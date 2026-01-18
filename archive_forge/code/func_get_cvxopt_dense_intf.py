import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def get_cvxopt_dense_intf():
    """Dynamic import of CVXOPT dense interface.
    """
    import cvxpy.interface.cvxopt_interface.valuerix_interface as dmi
    return dmi.DenseMatrixInterface()