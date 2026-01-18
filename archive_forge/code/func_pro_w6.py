import re
from nltk.stem.api import StemmerI
def pro_w6(self, word):
    """process length six patterns and extract length three roots"""
    if word.startswith('است') or word.startswith('مست'):
        word = word[3:]
    elif word[0] == 'م' and word[3] == 'ا' and (word[5] == 'ة'):
        word = word[1:3] + word[4]
    elif word[0] == 'ا' and word[2] == 'ت' and (word[4] == 'ا'):
        word = word[1] + word[3] + word[5]
    elif word[0] == 'ا' and word[3] == 'و' and (word[2] == word[4]):
        word = word[1] + word[4:]
    elif word[0] == 'ت' and word[2] == 'ا' and (word[4] == 'ي'):
        word = word[1] + word[3] + word[5]
    else:
        word = self.suf1(word)
        if len(word) == 6:
            word = self.pre1(word)
    return word