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
def pseudorandom_weak_vector(size, seed_string=None, hashfxn=hash):
    """Get a random vector, derived deterministically from `seed_string` if supplied.

    Useful for initializing KeyedVectors that will be the starting projection/input layers of _2Vec models.

    """
    if seed_string:
        once = np.random.Generator(np.random.SFC64(hashfxn(seed_string) & 4294967295))
    else:
        once = utils.default_prng
    return (once.random(size).astype(REAL) - 0.5) / size