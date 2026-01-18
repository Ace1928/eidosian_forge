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
def scipy2sparse(vec, eps=1e-09):
    """Convert a scipy.sparse vector into the Gensim bag-of-words format.

    Parameters
    ----------
    vec : `scipy.sparse`
        Sparse vector.

    eps : float, optional
        Value used for threshold, all coordinates less than `eps` will not be presented in result.

    Returns
    -------
    list of (int, float)
        Vector in Gensim bag-of-words format.

    """
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return [(int(pos), float(val)) for pos, val in zip(vec.indices, vec.data) if np.abs(val) > eps]