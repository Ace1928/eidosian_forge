from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return np.all((x >= lb) & (x <= ub))