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
def test_bounds_p0(self):

    def f(x, a):
        return np.sin(x + a)
    xdata = np.linspace(-2 * np.pi, 2 * np.pi, 40)
    ydata = np.sin(xdata)
    bounds = (-3 * np.pi, 3 * np.pi)
    for method in ['trf', 'dogbox']:
        popt_1, _ = curve_fit(f, xdata, ydata, p0=2.1 * np.pi)
        popt_2, _ = curve_fit(f, xdata, ydata, p0=2.1 * np.pi, bounds=bounds, method=method)
        assert_allclose(popt_1, popt_2)