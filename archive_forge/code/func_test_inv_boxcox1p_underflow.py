import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_inv_boxcox1p_underflow():
    x = 1e-15
    lam = 1e-306
    y = inv_boxcox1p(x, lam)
    assert_allclose(y, x, rtol=1e-14)