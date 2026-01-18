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
def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
    return A0 * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2)) + A1 * np.exp(-(x - x1) ** 2 / (2.0 * sigma ** 2)) + c