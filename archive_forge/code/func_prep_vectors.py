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
def prep_vectors(target_shape, prior_vectors=None, seed=0, dtype=REAL):
    """Return a numpy array of the given shape. Reuse prior_vectors object or values
    to extent possible. Initialize new values randomly if requested.

    """
    if prior_vectors is None:
        prior_vectors = np.zeros((0, 0))
    if prior_vectors.shape == target_shape:
        return prior_vectors
    target_count, vector_size = target_shape
    rng = np.random.default_rng(seed=seed)
    new_vectors = rng.random(target_shape, dtype=dtype)
    new_vectors *= 2.0
    new_vectors -= 1.0
    new_vectors /= vector_size
    new_vectors[0:prior_vectors.shape[0], 0:prior_vectors.shape[1]] = prior_vectors
    return new_vectors