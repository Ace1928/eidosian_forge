import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_dense2(self):
    A = np.eye(6)
    A[-2, -1] = 1
    A[-1, :] = 1
    b = np.zeros(A.shape[0])
    A1, b1, status, message = self.rr(A, b)
    assert_(redundancy_removed(A1, A))
    assert_equal(status, 0)