import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_m_gt_n(self):
    np.random.seed(2032)
    m, n = (20, 10)
    A0 = np.random.rand(m, n)
    b0 = np.random.rand(m)
    x = np.linalg.solve(A0[:n, :], b0[:n])
    b0[n:] = A0[n:, :].dot(x)
    A1, b1, status, message = self.rr(A0, b0)
    assert_equal(status, 0)
    assert_equal(A1.shape[0], n)
    assert_equal(np.linalg.matrix_rank(A1), n)