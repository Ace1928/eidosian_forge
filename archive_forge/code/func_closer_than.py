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
def closer_than(self, key1, key2):
    """Get all keys that are closer to `key1` than `key2` is to `key1`."""
    all_distances = self.distances(key1)
    e1_index = self.get_index(key1)
    e2_index = self.get_index(key2)
    closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
    return [self.index_to_key[index] for index in closer_node_indices if index != e1_index]