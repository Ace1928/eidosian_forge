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
def full2sparse_clipped(vec, topn, eps=1e-09):
    """Like :func:`~gensim.matutils.full2sparse`, but only return the `topn` elements of the greatest magnitude (abs).

    This is more efficient that sorting a vector and then taking the greatest values, especially
    where `len(vec) >> topn`.

    Parameters
    ----------
    vec : numpy.ndarray
        Input dense vector
    topn : int
        Number of greatest (abs) elements that will be presented in result.
    eps : float
        Threshold value, if coordinate in `vec` < eps, this will not be presented in result.

    Returns
    -------
    list of (int, float)
        Clipped vector in BoW format.

    See Also
    --------
    :func:`~gensim.matutils.full2sparse`
        Convert dense array to gensim bag-of-words format.

    """
    if topn <= 0:
        return []
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    biggest = nnz.take(argsort(abs(vec).take(nnz), topn, reverse=True))
    return list(zip(biggest, vec.take(biggest)))