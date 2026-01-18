import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_infeasible_zero_row(self):
    A = np.eye(3)
    A[1, :] = 0
    b = np.random.rand(3)
    A1, b1, status, message = self.rr(A, b)
    assert_equal(status, 2)