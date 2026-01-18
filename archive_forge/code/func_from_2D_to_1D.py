import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def from_2D_to_1D(constant):
    """Convert 2D Numpy matrices or arrays to 1D.
    """
    if isinstance(constant, np.ndarray) and constant.ndim == 2:
        return np.asarray(constant)[:, 0]
    else:
        return constant