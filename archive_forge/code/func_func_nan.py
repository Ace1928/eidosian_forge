import numpy as np
from statsmodels.tools.rootfinding import brentq_expanding
from numpy.testing import (assert_allclose, assert_equal, assert_raises,
def func_nan(x, a, b):
    x = np.atleast_1d(x)
    f = (x - 1.0 * a) ** 3
    f[x < b] = np.nan
    return f