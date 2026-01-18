from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def compute_grad(J, f):
    """Compute gradient of the least-squares cost function."""
    if isinstance(J, LinearOperator):
        return J.rmatvec(f)
    else:
        return J.T.dot(f)