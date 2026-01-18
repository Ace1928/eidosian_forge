import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
def test_boxcox_underflow():
    x = 1 + 1e-15
    lmbda = 1e-306
    y = boxcox(x, lmbda)
    assert_allclose(y, np.log(x), rtol=1e-14)