import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_mad_axis_none():
    a = np.array([[0, 1, 2], [2, 3, 2]])

    def m(x):
        return np.median(x)
    direct = mad(a=a, axis=None)
    custom = mad(a=a, axis=None, center=m)
    axis0 = mad(a=a.ravel(), axis=0)
    np.testing.assert_allclose(direct, custom)
    np.testing.assert_allclose(direct, axis0)