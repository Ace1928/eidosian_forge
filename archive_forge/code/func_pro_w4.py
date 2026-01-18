import re
from nltk.stem.api import StemmerI
def pro_w4(self, word):
    """process length four patterns and extract length three roots"""
    if word[0] in self.pr4[0]:
        word = word[1:]
    elif word[1] in self.pr4[1]:
        word = word[:1] + word[2:]
    elif word[2] in self.pr4[2]:
        word = word[:2] + word[3]
    elif word[3] in self.pr4[3]:
        word = word[:-1]
    else:
        word = self.suf1(word)
        if len(word) == 4:
            word = self.pre1(word)
    return word