import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_inv_boxcox():
    x = np.array([0.0, 1.0, 2.0])
    lam = np.array([0.0, 1.0, 2.0])
    y = boxcox(x, lam)
    x2 = inv_boxcox(y, lam)
    assert_almost_equal(x, x2)
    x = np.array([0.0, 1.0, 2.0])
    lam = np.array([0.0, 1.0, 2.0])
    y = boxcox1p(x, lam)
    x2 = inv_boxcox1p(y, lam)
    assert_almost_equal(x, x2)