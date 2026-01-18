import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def get_co_occurrences(self, word1, word2):
    """Return number of docs the words co-occur in, once `accumulate` has been called."""
    raise NotImplementedError('Word2Vec model does not support co-occurrence counting')