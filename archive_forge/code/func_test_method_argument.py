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
def test_method_argument(self):

    def f(x, a, b):
        return a * np.exp(-b * x)
    xdata = np.linspace(0, 1, 11)
    ydata = f(xdata, 2.0, 2.0)
    for method in ['trf', 'dogbox', 'lm', None]:
        popt, pcov = curve_fit(f, xdata, ydata, method=method)
        assert_allclose(popt, [2.0, 2.0])
    assert_raises(ValueError, curve_fit, f, xdata, ydata, method='unknown')