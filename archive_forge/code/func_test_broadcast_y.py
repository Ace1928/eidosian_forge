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
def test_broadcast_y(self):
    xdata = np.arange(10)
    target = 4.7 * xdata ** 2 + 3.5 * xdata + np.random.rand(len(xdata))

    def fit_func(x, a, b):
        return a * x ** 2 + b * x - target
    for method in ['lm', 'trf', 'dogbox']:
        popt0, pcov0 = curve_fit(fit_func, xdata=xdata, ydata=np.zeros_like(xdata), method=method)
        popt1, pcov1 = curve_fit(fit_func, xdata=xdata, ydata=0, method=method)
        assert_allclose(pcov0, pcov1)