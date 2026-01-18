import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations
    of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Size of vocabulary.
    min_count: int
        Minimum collocation count threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Returns
    -------
    float
        Score for given phrase. Can be negative.

    Notes
    -----
    Formula: :math:`\\frac{(bigram\\_count - min\\_count) * len\\_vocab }{ (worda\\_count * wordb\\_count)}`.

    """
    denom = worda_count * wordb_count
    if denom == 0:
        return NEGATIVE_INFINITY
    return (bigram_count - min_count) / float(denom) * len_vocab