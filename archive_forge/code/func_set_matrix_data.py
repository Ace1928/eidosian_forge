from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def set_matrix_data(linC, linPy) -> None:
    """Calls the appropriate cvxcore function to set the matrix data field of
       our C++ linOp.
    """
    if get_type(linPy) == cvxcore.SPARSE_CONST:
        coo = format_matrix(linPy.data, format='sparse')
        linC.set_sparse_data(coo.data, coo.row.astype(float), coo.col.astype(float), coo.shape[0], coo.shape[1])
    else:
        linC.set_dense_data(format_matrix(linPy.data, shape=linPy.shape))
        linC.set_data_ndim(len(linPy.data.shape))