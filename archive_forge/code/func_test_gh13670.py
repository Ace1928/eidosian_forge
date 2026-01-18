import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_gh13670(self):
    rng = np.random.default_rng(8250058582555444926)
    x = np.linspace(0, 3, 101)
    y = 2 * x + 1 + rng.normal(size=101) * 0.5

    def line(x, *p):
        assert not np.all(line.last_p == p)
        line.last_p = p
        return x * p[0] + p[1]

    def jac(x, *p):
        assert not np.all(jac.last_p == p)
        jac.last_p = p
        return np.array([x, np.ones_like(x)]).T
    line.last_p = None
    jac.last_p = None
    p0 = np.array([1.0, 5.0])
    curve_fit(line, x, y, p0, method='lm', jac=jac)