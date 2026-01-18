import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_compare_dense_and_sparse(self):
    D = np.diag(range(1, 101))
    A = np.hstack([D, D, D, D])
    A_sparse = csc_matrix(A)
    np.random.seed(0)
    Z, LS, Y = projections(A)
    Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)
    for k in range(20):
        z = np.random.normal(size=(400,))
        assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))
        assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))
        x = np.random.normal(size=(100,))
        assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))