import numpy as np
from numpy.linalg import lstsq, norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult
from .common import (
def lsmr_operator(Jop, d, active_set):
    """Compute LinearOperator to use in LSMR by dogbox algorithm.

    `active_set` mask is used to excluded active variables from computations
    of matrix-vector products.
    """
    m, n = Jop.shape

    def matvec(x):
        x_free = x.ravel().copy()
        x_free[active_set] = 0
        return Jop.matvec(x * d)

    def rmatvec(x):
        r = d * Jop.rmatvec(x)
        r[active_set] = 0
        return r
    return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)