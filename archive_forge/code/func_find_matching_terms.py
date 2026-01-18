import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
@staticmethod
def find_matching_terms(trie, word):
    """Find terms in the `trie` beginning with the `word`.

        Parameters
        ----------
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching terms.
        word : str
            Input word to use for prefix search.

        Returns
        -------
        list of str
            List of matching terms.

        """
    matches = trie.items('%s.' % word)
    matching_terms = [''.join(key_chars) for key_chars, value in matches]
    return matching_terms