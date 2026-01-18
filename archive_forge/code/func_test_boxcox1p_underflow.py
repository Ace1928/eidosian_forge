import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox1p_underflow():
    x = np.array([1e-15, 1e-306])
    lmbda = np.array([1e-306, 1e-18])
    y = boxcox1p(x, lmbda)
    assert_allclose(y, np.log1p(x), rtol=1e-14)