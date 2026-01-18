from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def set_linC_data(linC, linPy) -> None:
    """Sets numerical data fields in linC."""
    assert linPy.data is not None
    if isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
        set_slice_data(linC, linPy)
    elif isinstance(linPy.data, float) or isinstance(linPy.data, numbers.Integral):
        linC.set_dense_data(format_matrix(linPy.data, format='scalar'))
        linC.set_data_ndim(0)
    else:
        set_matrix_data(linC, linPy)