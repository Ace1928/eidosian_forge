import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_m_lt_n_rank_deficient(self):
    m, n = (9, 10)
    A0 = np.random.rand(m, n)
    b0 = np.random.rand(m)
    A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
    b0[-1] = np.arange(m - 1).dot(b0[:-1])
    A1, b1, status, message = self.rr(A0, b0)
    assert_equal(status, 0)
    assert_equal(A1.shape[0], 8)
    assert_equal(np.linalg.matrix_rank(A1), 8)