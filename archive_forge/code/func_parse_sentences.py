from nltk.corpus import wordnet
from nltk.wsd import lesk
import spacy
def parse_sentences(sentences, nlp_tool):
    return [nlp_tool(sentence) for sentence in sentences]