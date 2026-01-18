from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def nonzero_csc_matrix(A):
    assert not np.isnan(A.data).any()
    zero_indices = A.data == 0
    A.data[zero_indices] = np.nan
    A_rows, A_cols = A.nonzero()
    ind = np.argsort(A_cols, kind='mergesort')
    A_rows = A_rows[ind]
    A_cols = A_cols[ind]
    A.data[zero_indices] = 0
    return (A_rows, A_cols)