from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def inside_box_boundaries(x, lb, ub):
    """Check if lb <= x <= ub."""
    return (lb <= x).all() and (x <= ub).all()