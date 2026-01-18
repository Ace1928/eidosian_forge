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
@pytest.mark.parametrize('method', ['lm', 'trf', 'dogbox'])
def test_nan_policy_1d(self, method):

    def f(x, a, b):
        return a * x + b
    xdata_with_nan = np.array([2, 3, np.nan, 4, 4, np.nan])
    ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7])
    xdata_without_nan = np.array([2, 3, 4])
    ydata_without_nan = np.array([1, 2, 3])
    self._check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method)