import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def similar_by_word(self, word, topn=10, restrict_vocab=None):
    """Compatibility alias for similar_by_key()."""
    return self.similar_by_key(word, topn, restrict_vocab)