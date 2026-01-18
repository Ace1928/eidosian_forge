import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
def test_no_redundancy(self):
    m, n = (10, 10)
    A0 = np.random.rand(m, n)
    b0 = np.random.rand(m)
    A1, b1, status, message = self.rr(A0, b0)
    assert_allclose(A0, A1)
    assert_allclose(b0, b1)
    assert_equal(status, 0)