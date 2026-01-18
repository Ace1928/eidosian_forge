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
def sort_by_descending_frequency(self):
    """Sort the vocabulary so the most frequent words have the lowest indexes."""
    if not len(self):
        return
    count_sorted_indexes = np.argsort(self.expandos['count'])[::-1]
    self.index_to_key = [self.index_to_key[idx] for idx in count_sorted_indexes]
    self.allocate_vecattrs()
    for k in self.expandos:
        self.expandos[k] = self.expandos[k][count_sorted_indexes]
    if len(self.vectors):
        logger.warning('sorting after vectors have been allocated is expensive & error-prone')
        self.vectors = self.vectors[count_sorted_indexes]
    self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}