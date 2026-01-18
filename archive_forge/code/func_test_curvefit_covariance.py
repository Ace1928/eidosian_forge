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
def test_curvefit_covariance(self):

    def funcp(x, a, b):
        rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
        return rotn.dot(a * np.exp(-b * x))

    def jacp(x, a, b):
        rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
        e = np.exp(-b * x)
        return rotn.dot(np.vstack((e, -a * x * e)).T)

    def func(x, a, b):
        return a * np.exp(-b * x)

    def jac(x, a, b):
        e = np.exp(-b * x)
        return np.vstack((e, -a * x * e)).T
    np.random.seed(0)
    xdata = np.arange(1, 4)
    y = func(xdata, 2.5, 1.0)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))
    sigma = np.zeros(len(xdata)) + 0.2
    covar = np.diag(sigma ** 2)
    rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
    ydatap = rotn.dot(ydata)
    covarp = rotn.dot(covar).dot(rotn.T)
    for jac1, jac2 in [(jac, jacp), (None, None)]:
        for absolute_sigma in [False, True]:
            popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma, jac=jac1, absolute_sigma=absolute_sigma)
            popt2, pcov2 = curve_fit(funcp, xdata, ydatap, sigma=covarp, jac=jac2, absolute_sigma=absolute_sigma)
            assert_allclose(popt1, popt2, rtol=1.2e-07, atol=1e-14)
            assert_allclose(pcov1, pcov2, rtol=1.2e-07, atol=1e-14)