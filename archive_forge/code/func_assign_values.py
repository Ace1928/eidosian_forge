import re
import warnings
from string import punctuation
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams
def assign_values(self, token):
    """
        Assigns each phoneme its value from the sonority hierarchy.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return: List of tuples, first element is character/phoneme and
                 second is the soronity value.
        :rtype: list(tuple(str, int))
        """
    syllables_values = []
    for c in token:
        try:
            syllables_values.append((c, self.phoneme_map[c]))
        except KeyError:
            if c not in '0123456789' and c not in punctuation:
                warnings.warn("Character not defined in sonority_hierarchy, assigning as vowel: '{}'".format(c))
                syllables_values.append((c, max(self.phoneme_map.values())))
                if c not in self.vowels:
                    self.vowels += c
            else:
                syllables_values.append((c, -1))
    return syllables_values