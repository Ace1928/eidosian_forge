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
def test_indeterminate_covariance(self):
    xdata = np.array([1, 2, 3, 4, 5, 6])
    ydata = np.array([1, 2, 3, 4, 5.5, 6])
    assert_warns(OptimizeWarning, curve_fit, lambda x, a, b: a * x, xdata, ydata)