from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def compute_jac_scale(J, scale_inv_old=None):
    """Compute variables scale based on the Jacobian matrix."""
    if issparse(J):
        scale_inv = np.asarray(J.power(2).sum(axis=0)).ravel() ** 0.5
    else:
        scale_inv = np.sum(J ** 2, axis=0) ** 0.5
    if scale_inv_old is None:
        scale_inv[scale_inv == 0] = 1
    else:
        scale_inv = np.maximum(scale_inv, scale_inv_old)
    return (1 / scale_inv, scale_inv)