import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, hstack
import pytest
def test_csr_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)
    for i in range(N):
        arr_col = X[:, i:i + 1]
        csr_col = Xcsr.getcol(i)
        assert_array_almost_equal(arr_col, csr_col.toarray())
        assert_(type(csr_col) is csr_matrix)