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
def test_array_trivial(self):

    def func(x):
        return 2.0 * x
    x0 = [0.3, 0.15]
    with np.errstate(all='ignore'):
        x = fixed_point(func, x0)
    assert_almost_equal(x, [0.0, 0.0])