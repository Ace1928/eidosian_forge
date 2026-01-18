import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
def hesstrue(self, params):
    return None
    y, x = (self.y, self.x)
    return -x * 2 * (y - np.dot(x, params))[:, None]