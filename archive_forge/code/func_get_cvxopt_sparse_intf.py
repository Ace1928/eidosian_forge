import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def get_cvxopt_sparse_intf():
    """Dynamic import of CVXOPT sparse interface.
    """
    import cvxpy.interface.cvxopt_interface.sparse_matrix_interface as smi
    return smi.SparseMatrixInterface()