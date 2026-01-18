import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_infs():
    h, a = (0, np.inf)
    res = 1 / (2 * np.pi) * np.arctan(a)
    assert_allclose(sc.owens_t(h, a), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -a), -res, rtol=5e-14)
    h = 1
    res = 0.07932762696572854
    assert_allclose(sc.owens_t(h, np.inf), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -np.inf), -res, rtol=5e-14)
    assert_equal(sc.owens_t(np.inf, 1), 0)
    assert_equal(sc.owens_t(-np.inf, 1), 0)
    assert_equal(sc.owens_t(np.inf, np.inf), 0)
    assert_equal(sc.owens_t(-np.inf, np.inf), 0)
    assert_equal(sc.owens_t(np.inf, -np.inf), -0.0)
    assert_equal(sc.owens_t(-np.inf, -np.inf), -0.0)