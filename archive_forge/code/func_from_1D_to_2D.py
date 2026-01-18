import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def from_1D_to_2D(constant):
    """Convert 1D Numpy arrays to matrices.
    """
    if isinstance(constant, np.ndarray) and constant.ndim == 1:
        return np.asmatrix(constant).T
    else:
        return constant