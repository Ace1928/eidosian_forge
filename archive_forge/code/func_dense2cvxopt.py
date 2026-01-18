import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def dense2cvxopt(value):
    """Converts a NumPy matrix to a CVXOPT matrix.

    Parameters
    ----------
    value : NumPy matrix/ndarray
        The matrix to convert.

    Returns
    -------
    CVXOPT matrix
        The converted matrix.
    """
    import cvxopt
    return cvxopt.matrix(value, tc='d')