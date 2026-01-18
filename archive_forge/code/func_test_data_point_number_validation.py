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
def test_data_point_number_validation(self):

    def func(x, a, b, c, d, e):
        return a * np.exp(-b * x) + c + d + e
    with assert_raises(TypeError, match='The number of func parameters='):
        curve_fit(func, xdata=[1, 2, 3, 4], ydata=[5, 9, 13, 17])