import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def cvxopt2dense(value):
    """Converts a CVXOPT matrix to a NumPy ndarray.

    Parameters
    ----------
    value : CVXOPT matrix
        The matrix to convert.

    Returns
    -------
    NumPy ndarray
        The converted matrix.
    """
    return np.array(value)