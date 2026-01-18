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
def test_curvefit_simplecovariance(self):

    def func(x, a, b):
        return a * np.exp(-b * x)

    def jac(x, a, b):
        e = np.exp(-b * x)
        return np.vstack((e, -a * x * e)).T
    np.random.seed(0)
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))
    sigma = np.zeros(len(xdata)) + 0.2
    covar = np.diag(sigma ** 2)
    for jac1, jac2 in [(jac, jac), (None, None)]:
        for absolute_sigma in [False, True]:
            popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma, jac=jac1, absolute_sigma=absolute_sigma)
            popt2, pcov2 = curve_fit(func, xdata, ydata, sigma=covar, jac=jac2, absolute_sigma=absolute_sigma)
            assert_allclose(popt1, popt2, atol=1e-14)
            assert_allclose(pcov1, pcov2, atol=1e-14)