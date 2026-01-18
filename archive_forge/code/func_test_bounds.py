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
def test_bounds(self):

    def f(x, a, b):
        return a * np.exp(-b * x)
    xdata = np.linspace(0, 1, 11)
    ydata = f(xdata, 2.0, 2.0)
    lb = [1.0, 0]
    ub = [1.5, 3.0]
    bounds = (lb, ub)
    bounds_class = Bounds(lb, ub)
    for method in [None, 'trf', 'dogbox']:
        popt, pcov = curve_fit(f, xdata, ydata, bounds=bounds, method=method)
        assert_allclose(popt[0], 1.5)
        popt_class, pcov_class = curve_fit(f, xdata, ydata, bounds=bounds_class, method=method)
        assert_allclose(popt_class, popt)
    popt, pcov = curve_fit(f, xdata, ydata, method='trf', bounds=([0.0, 0], [0.6, np.inf]))
    assert_allclose(popt[0], 0.6)
    assert_raises(ValueError, curve_fit, f, xdata, ydata, bounds=bounds, method='lm')