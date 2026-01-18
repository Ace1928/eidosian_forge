from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def sparse_jacobian_csc(self, exprs, dep):
    """ Wraps Matrix/ndarray around results of .util.sparse_jacobian_csc """
    jac_exprs, colptrs, rowvals = sparse_jacobian_csc(exprs, dep)
    nnz = len(jac_exprs)
    return (self.Matrix(1, nnz, jac_exprs), np.asarray(colptrs, dtype=int), np.asarray(rowvals, dtype=int))