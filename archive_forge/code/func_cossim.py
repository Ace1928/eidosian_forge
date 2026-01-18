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
def cossim(vec1, vec2):
    """Get cosine similarity between two sparse vectors.

    Cosine similarity is a number between `<-1.0, 1.0>`, higher means more similar.

    Parameters
    ----------
    vec1 : list of (int, float)
        Vector in BoW format.
    vec2 : list of (int, float)
        Vector in BoW format.

    Returns
    -------
    float
        Cosine similarity between `vec1` and `vec2`.

    """
    vec1, vec2 = (dict(vec1), dict(vec2))
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum((val * val for val in vec1.values())))
    vec2len = 1.0 * math.sqrt(sum((val * val for val in vec2.values())))
    assert vec1len > 0.0 and vec2len > 0.0, 'sparse documents must not contain any explicit zero entries'
    if len(vec2) < len(vec1):
        vec1, vec2 = (vec2, vec1)
    result = sum((value * vec2.get(index, 0.0) for index, value in vec1.items()))
    result /= vec1len * vec2len
    return result