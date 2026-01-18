import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox_basic():
    x = np.array([0.5, 1, 2, 4])
    y = boxcox(x, 0)
    assert_almost_equal(y, np.log(x))
    y = boxcox(x, 1)
    assert_almost_equal(y, x - 1)
    y = boxcox(x, 2)
    assert_almost_equal(y, 0.5 * (x ** 2 - 1))
    lam = np.array([0.5, 1, 2])
    y = boxcox(0, lam)
    assert_almost_equal(y, -1.0 / lam)