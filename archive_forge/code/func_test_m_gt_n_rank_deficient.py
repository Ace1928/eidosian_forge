import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_m_gt_n_rank_deficient(self):
    m, n = (20, 10)
    A0 = np.zeros((m, n))
    A0[:, 0] = 1
    b0 = np.ones(m)
    A1, b1, status, message = self.rr(A0, b0)
    assert_equal(status, 0)
    assert_allclose(A1, A0[0:1, :])
    assert_allclose(b1, b0[0])