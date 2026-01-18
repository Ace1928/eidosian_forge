import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from .givens_elimination import givens_elimination
from .common import (
def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
    """Solve regularized least squares using information from QR-decomposition.

    The initial problem is to solve the following system in a least-squares
    sense::

        A x = b
        D x = 0

    where D is diagonal matrix. The method is based on QR decomposition
    of the form A P = Q R, where P is a column permutation matrix, Q is an
    orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    m, n : int
        Initial shape of A.
    R : ndarray, shape (n, n)
        Upper triangular matrix from QR decomposition of A.
    QTb : ndarray, shape (n,)
        First n components of Q^T b.
    perm : ndarray, shape (n,)
        Array defining column permutation of A, such that ith column of
        P is perm[i]-th column of identity matrix.
    diag : ndarray, shape (n,)
        Array containing diagonal elements of D.

    Returns
    -------
    x : ndarray, shape (n,)
        Found least-squares solution.
    """
    if copy_R:
        R = R.copy()
    v = QTb.copy()
    givens_elimination(R, v, diag[perm])
    abs_diag_R = np.abs(np.diag(R))
    threshold = EPS * max(m, n) * np.max(abs_diag_R)
    nns, = np.nonzero(abs_diag_R > threshold)
    R = R[np.ix_(nns, nns)]
    v = v[nns]
    x = np.zeros(n)
    x[perm[nns]] = solve_triangular(R, v)
    return x