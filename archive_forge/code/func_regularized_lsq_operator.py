from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def regularized_lsq_operator(J, diag):
    """Return a matrix arising in regularized least squares as LinearOperator.

    The matrix is
        [ J ]
        [ D ]
    where D is diagonal matrix with elements from `diag`.
    """
    J = aslinearoperator(J)
    m, n = J.shape

    def matvec(x):
        return np.hstack((J.matvec(x), diag * x))

    def rmatvec(x):
        x1 = x[:m]
        x2 = x[m:]
        return J.rmatvec(x1) + diag * x2
    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec)