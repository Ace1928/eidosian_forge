import re
from nltk.stem.api import StemmerI
def end_w6(self, word):
    """ending step (word of length six)"""
    if len(word) == 5:
        word = self.pro_w53(word)
        word = self.end_w5(word)
    elif len(word) == 6:
        word = self.pro_w64(word)
    return word