import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_dense1(self):
    A = np.ones((6, 6))
    A[0, :3] = 0
    A[1, 3:] = 0
    A[3:, ::2] = -1
    A[3, :2] = 0
    A[4, 2:] = 0
    b = np.zeros(A.shape[0])
    A1, b1, status, message = self.rr(A, b)
    assert_(redundancy_removed(A1, A))
    assert_equal(status, 0)