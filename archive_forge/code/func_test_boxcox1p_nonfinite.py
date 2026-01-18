import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox1p_nonfinite():
    x = np.array([-2, -2, -1.5])
    y = boxcox1p(x, [0.5, 2.0, -1.5])
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))
    x = -1
    y = boxcox1p(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))