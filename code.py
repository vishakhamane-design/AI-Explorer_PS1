import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load AI model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load database (RAG)
try:
    db = pd.read_csv("certified_brands.csv")
except:
    db = pd.DataFrame(columns=["brand", "certification"])

# Buzzwords
buzzwords = ["eco-friendly", "green", "natural", "sustainable", "chemical-free"]

# 🌿 Streamlit page config
st.set_page_config(page_title="Sustainability Auditor", layout="centered")

# 🌱 Inject CSS for blur background + styling
st.markdown("""
<style>
/* Full page background */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1470&q=80');
    background-size: cover;
    background-attachment: fixed;
}

/* Frosted glass effect for main container */
.block-container {
    backdrop-filter: blur(10px);
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 15px;
    padding: 2rem;
    color: black;
}

/* Buzzword highlight */
.buzzword {
    background-color: #a5d6a7;
    color: black;
    font-weight: bold;
    padding: 2px 4px;
    border-radius: 4px;
}

/* Certified brand highlight */
.certified {
    background-color: #64b5f6;
    color: black;
    font-weight: bold;
    padding: 2px 4px;
    border-radius: 4px;
}

/* Buttons */
div.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
    font-size: 16px;
}

/* Text area and input boxes */
.stTextInput, .stTextArea {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    color: black;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #81c784;
}
</style>
""", unsafe_allow_html=True)

# UI Elements
st.title("🌱 Intent-Aware Sustainability Auditor")
st.subheader("Check your products for greenwashing or sustainability claims!")

text = st.text_area("📝 Enter Product Description")
url = st.text_input("🔗 OR paste product URL")

if st.button("🔍 Analyze"):
    input_text = ""
    if text:
        input_text = text.lower()
    elif url:
        input_text = url.lower()
    else:
        st.warning("Please enter text or paste a URL")
        st.stop()

    # 🔴 Buzzwords
    found = [word for word in buzzwords if word in input_text]
    buzzwords_penalty = len(found) * 15

    # 🧠 AI Sentiment
    result = classifier(input_text)[0]
    label = result['label']
    confidence = result['score']
    ai_penalty = 20 if label == "NEGATIVE" else 0

    # 🟢 RAG Certification Match
    matched_brands = []
    cert_bonus = 0
    for brand in db['brand']:
        if brand.lower() in input_text:
            matched_brands.append(brand)
            cert_bonus = 20

    # 🎯 Final Score
    score = 100 - buzzwords_penalty - ai_penalty + cert_bonus
    score = max(0, min(score, 100))

    # 📊 Display results
    st.subheader("📊 Results")
    if score > 60:
        st.success("✅ Likely Evidence-Based")
    else:
        st.error("❌ Likely Greenwashing")
    st.write(f"🌱 Sustainability Score: {score}/100")
    st.progress(score/100)

    # 🌿 Multi-Bar Chart with gradient colors and aligned labels
    components = ["Buzzwords Penalty", "AI Sentiment Impact", "Certification Bonus", "Final Score"]
    values = [buzzwords_penalty, ai_penalty, cert_bonus, score]

    # Progressive colors based on value
    colors = []
    for v in values:
        if v <= 20:
            colors.append("#81c784")  # light green
        elif v <= 40:
            colors.append("#66bb6a")
        elif v <= 60:
            colors.append("#388e3c")
        elif v <= 80:
            colors.append("#2e7d32")
        else:
            colors.append("#1b5e20")  # dark green

    fig, ax = plt.subplots(figsize=(8,6))  # wider figure
    bars = ax.bar(components, values, color=colors, edgecolor="black")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score / Impact", fontsize=12, color="#2e7d32")
    ax.set_title("🌱 Sustainability Score Components", fontsize=14, color="#2e7d32")

    # Rotate x-axis labels for clarity
    plt.xticks(rotation=20, ha="right", fontsize=11)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{int(height)}", 
                ha='center', va='bottom', color="#2e7d32", fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Highlight buzzwords and certified brands in text
    display_text = input_text
    for word in buzzwords:
        display_text = display_text.replace(word, f'<span class="buzzword">{word}</span>')
    for brand in matched_brands:
        display_text = display_text.replace(brand.lower(), f'<span class="certified">{brand}</span>')

    st.markdown("### 🔍 Analyzed Text")
    st.markdown(display_text, unsafe_allow_html=True)

    # Buzzwords found
    st.write("🔴 Buzzwords Found:")
    st.write(found if found else "None")

    # Certifications
    st.write("🟢 Certifications Found:")
    if matched_brands:
        for brand in matched_brands:
            cert = db[db['brand'] == brand]['certification'].values[0]
            st.markdown(f"- ✅ {brand} ({cert})")
    else:
        st.write("None")

    # AI Confidence
    st.write(f"🤖 AI Confidence: {round(confidence*100,2)}%")

    # Reasoning
    st.subheader("🧠 Reasoning")
    if found and not matched_brands:
        st.write("⚠️ Uses vague marketing terms without proof.")
    elif matched_brands:
        st.write("✅ Contains verified certified brands.")
    else:
        st.write("ℹ️ Neutral claims based on AI analysis.")