import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def compute_banded_symmetric_XT_W_Y(X, w, Y):
    """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
        w : array, shape (n,)
            Array of weights
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.

        """
    W_Y = np.copy(Y)
    W_Y[2] *= w
    for i in range(2):
        W_Y[i, 2 - i:] *= w[:-2 + i]
        W_Y[3 + i, :-1 - i] *= w[1 + i:]
    n = X.shape[1]
    res = np.zeros((4, n))
    for i in range(n):
        for j in range(min(n - i, 4)):
            res[-j - 1, i + j] = sum(X[j:, i] * W_Y[:5 - j, i + j])
    return res