import numpy as np
from statsmodels.tools.rootfinding import brentq_expanding
from numpy.testing import (assert_allclose, assert_equal, assert_raises,
def funcn(x, a):
    f = -(x - a) ** 3
    return f