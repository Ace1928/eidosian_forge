import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def get_meaning_formula(self, generic, word):
    """
        :param generic: A meaning formula string containing the
            parameter "<word>"
        :param word: The actual word to be replace "<word>"
        """
    word = word.replace('.', '')
    return generic.replace('<word>', word)