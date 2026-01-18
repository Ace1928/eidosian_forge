from collections import Counter
from nltk.tokenize.api import TokenizerI
def onset(self, word):
    """
        Returns consonant cluster of word, i.e. all characters until the first vowel.

        :param word: Single word or token
        :type word: str
        :return: String of characters of onset
        :rtype: str
        """
    onset = ''
    for c in word.lower():
        if c in self.vowels:
            return onset
        else:
            onset += c
    return onset