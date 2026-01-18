from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def kullback_leibler(vec1, vec2, num_features=None):
    """Calculate Kullback-Leibler distance between two probability distributions using `scipy.stats.entropy`.

    Parameters
    ----------
    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    num_features : int, optional
        Number of features in the vectors.

    Returns
    -------
    float
        Kullback-Leibler distance between `vec1` and `vec2`.
        Value in range [0, +∞) where values closer to 0 mean less distance (higher similarity).

    """
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    return entropy(vec1, vec2)