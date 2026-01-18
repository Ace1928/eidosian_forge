import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox_nonfinite():
    x = np.array([-1, -1, -0.5])
    y = boxcox(x, [0.5, 2.0, -1.5])
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))
    x = 0
    y = boxcox(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))