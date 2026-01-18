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
def test_array_basic1(self):

    def func(x, c):
        return c * x ** 2
    c = array([0.75, 1.0, 1.25])
    x0 = [1.1, 1.15, 0.9]
    with np.errstate(all='ignore'):
        x = fixed_point(func, x0, args=(c,))
    assert_almost_equal(x, 1.0 / c)