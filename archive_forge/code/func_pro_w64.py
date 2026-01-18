import re
from nltk.stem.api import StemmerI
def pro_w64(self, word):
    """process length six patterns and extract length four roots"""
    if word[0] == 'ا' and word[4] == 'ا':
        word = word[1:4] + word[5]
    elif word.startswith('مت'):
        word = word[2:]
    return word