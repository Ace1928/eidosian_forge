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
def create_vocab_trie(embedding):
    """Create trie with vocab terms of the given embedding to enable quick prefix searches.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which trie is to be created.

        Returns
        -------
        :class:`pygtrie.Trie`
            Trie containing vocab terms of the input embedding.

        """
    try:
        from pygtrie import Trie
    except ImportError:
        raise ImportError('pygtrie could not be imported, please install pygtrie in order to use LexicalEntailmentEvaluation')
    vocab_trie = Trie()
    for key in embedding.key_to_index:
        vocab_trie[key] = True
    return vocab_trie