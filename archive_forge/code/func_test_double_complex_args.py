import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
def test_double_complex_args(self):
    cx = np.array([0.5 + 1j, 0.25 - 0.375j, 13.0 - 4j], np.complex128)
    cy = np.array([0.875 + 2j, 0.875 - 0.625j, -1.0 + 2j], np.complex128)
    assert_equal(blas._test_izamax(cx), 3)
    assert_allclose(blas._test_zdotc(cx, cy), -18.109375 + 22.296875j)
    assert_allclose(blas._test_zdotu(cx, cy), -6.578125 + 31.390625j)
    assert_allclose(blas._test_zdotc(cx[::2], cy[::2]), -18.5625 + 22.125j)
    assert_allclose(blas._test_zdotu(cx[::2], cy[::2]), -6.5625 + 31.875j)