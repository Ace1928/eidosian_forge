import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox1p_basic():
    x = np.array([-0.25, -1e-20, 0, 1e-20, 0.25, 1, 3])
    y = boxcox1p(x, 0)
    assert_almost_equal(y, np.log1p(x))
    y = boxcox1p(x, 1)
    assert_almost_equal(y, x)
    y = boxcox1p(x, 2)
    assert_almost_equal(y, 0.5 * x * (2 + x))
    lam = np.array([0.5, 1, 2])
    y = boxcox1p(-1, lam)
    assert_almost_equal(y, -1.0 / lam)