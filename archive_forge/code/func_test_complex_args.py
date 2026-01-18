import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
def test_complex_args(self):
    cx = np.array([0.5 + 1j, 0.25 - 0.375j, 12.5 - 4j], np.complex64)
    cy = np.array([0.8 + 2j, 0.875 - 0.625j, -1.0 + 2j], np.complex64)
    assert_allclose(blas._test_cdotc(cx, cy), -17.6468753815 + 21.3718757629j)
    assert_allclose(blas._test_cdotu(cx, cy), -6.11562538147 + 30.3156242371j)
    assert_equal(blas._test_icamax(cx), 3)
    assert_allclose(blas._test_scasum(cx), 18.625)
    assert_allclose(blas._test_scnrm2(cx), 13.1796483994)
    assert_allclose(blas._test_cdotc(cx[::2], cy[::2]), -18.1000003815 + 21.2000007629j)
    assert_allclose(blas._test_cdotu(cx[::2], cy[::2]), -6.10000038147 + 30.7999992371j)
    assert_allclose(blas._test_scasum(cx[::2]), 18.0)
    assert_allclose(blas._test_scnrm2(cx[::2]), 13.1719398499)