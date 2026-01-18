from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def sparse_jacobian_csr(self, exprs, dep):
    """ Wraps Matrix/ndarray around results of .util.sparse_jacobian_csr """
    jac_exprs, rowptrs, colvals = sparse_jacobian_csr(exprs, dep)
    nnz = len(jac_exprs)
    return (self.Matrix(1, nnz, jac_exprs), np.asarray(rowptrs, dtype=int), np.asarray(colvals, dtype=int))