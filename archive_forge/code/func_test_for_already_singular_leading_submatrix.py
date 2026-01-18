import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
def test_for_already_singular_leading_submatrix(self):
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    cholesky, = get_lapack_funcs(('potrf',), (A,))
    c, k = cholesky(A, lower=False, overwrite_a=False, clean=True)
    delta, v = singular_leading_submatrix(A, c, k)
    A[k - 1, k - 1] += delta
    assert_array_almost_equal(det(A[:k, :k]), 0)
    quadratic_term = np.dot(v, np.dot(A, v))
    assert_array_almost_equal(quadratic_term, 0)