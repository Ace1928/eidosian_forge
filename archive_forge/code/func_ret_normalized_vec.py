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
def ret_normalized_vec(vec, length):
    """Normalize a vector in L2 (Euclidean unit norm).

    Parameters
    ----------
    vec : list of (int, number)
        Input vector in BoW format.
    length : float
        Length of vector

    Returns
    -------
    list of (int, number)
        L2-normalized vector in BoW format.

    """
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)