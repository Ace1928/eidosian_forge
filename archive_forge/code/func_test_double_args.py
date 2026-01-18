import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
def test_double_args(self):
    x = np.array([5.0, -3, -0.5], np.float64)
    y = np.array([2, 1, 0.5], np.float64)
    assert_allclose(blas._test_dasum(x), 8.5)
    assert_allclose(blas._test_ddot(x, y), 6.75)
    assert_allclose(blas._test_dnrm2(x), 5.85234975815)
    assert_allclose(blas._test_dasum(x[::2]), 5.5)
    assert_allclose(blas._test_ddot(x[::2], y[::2]), 9.75)
    assert_allclose(blas._test_dnrm2(x[::2]), 5.0249376297)
    assert_equal(blas._test_idamax(x), 1)