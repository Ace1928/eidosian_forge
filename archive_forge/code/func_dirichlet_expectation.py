import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def dirichlet_expectation(alpha):
    """For a vector :math:`\\theta \\sim Dir(\\alpha)`, compute :math:`E[log \\theta]`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.

    Returns
    -------
    numpy.ndarray:
        :math:`E[log \\theta]`

    """
    if len(alpha.shape) == 1:
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype, copy=False)