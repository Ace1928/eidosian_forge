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
def test_gh4555b(self):
    rng = np.random.default_rng(408113519974467917)

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise
    _, res = curve_fit(func, xdata, ydata)
    ref = [[+0.0158972536486215, 0.0069207183284242, -0.0007474400714749], [+0.0069207183284242, 0.0205057958128679, +0.0053997711275403], [-0.0007474400714749, 0.0053997711275403, +0.0027833930320877]]
    assert_allclose(res, ref, 2e-07)