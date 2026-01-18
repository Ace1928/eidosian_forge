from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def reinforce_box_boundaries(x, lb, ub):
    """Return clipped value of x"""
    return np.minimum(np.maximum(x, lb), ub)