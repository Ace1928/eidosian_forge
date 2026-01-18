import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_callback_shouldnt_affect_minimization(self):

    def callback(x):
        pass
    fun = optimize.rosen
    bounds = [(0, 10)] * 4
    x0 = [1, 2, 3, 4.0]
    res = optimize.minimize(fun, x0, bounds=bounds, method='TNC', options={'maxfun': 1000})
    res2 = optimize.minimize(fun, x0, bounds=bounds, method='TNC', options={'maxfun': 1000}, callback=callback)
    assert_allclose(res2.x, res.x)
    assert_allclose(res2.fun, res.fun)
    assert_equal(res2.nfev, res.nfev)