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
def jensen_shannon(vec1, vec2, num_features=None):
    """Calculate Jensen-Shannon distance between two probability distributions using `scipy.stats.entropy`.

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
        Jensen-Shannon distance between `vec1` and `vec2`.

    Notes
    -----
    This is a symmetric and finite "version" of :func:`gensim.matutils.kullback_leibler`.

    """
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))