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
def most_similar_to_given(self, key1, keys_list):
    """Get the `key` from `keys_list` most similar to `key1`."""
    return keys_list[argmax([self.similarity(key1, key) for key in keys_list])]