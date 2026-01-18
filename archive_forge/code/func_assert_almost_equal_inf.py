import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
def assert_almost_equal_inf(x, y, decimal=6, msg=None):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    assert_equal(np.isposinf(x), np.isposinf(y))
    assert_equal(np.isneginf(x), np.isneginf(y))
    assert_equal(np.isnan(x), np.isnan(y))
    assert_almost_equal(x[np.isfinite(x)], y[np.isfinite(y)])