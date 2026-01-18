import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
@pytest.mark.parametrize('x_lower, x_upper, y_lower, y_upper, expected', [(-np.inf, 0, -np.inf, 0, np.pi / 4), (-np.inf, -1, -np.inf, 0, np.pi / 4 * erfc(1)), (-np.inf, 0, -np.inf, -1, np.pi / 4 * erfc(1)), (-np.inf, -1, -np.inf, -1, np.pi / 4 * erfc(1) ** 2), (-np.inf, 1, -np.inf, 0, np.pi / 4 * (erf(1) + 1)), (-np.inf, 0, -np.inf, 1, np.pi / 4 * (erf(1) + 1)), (-np.inf, 1, -np.inf, 1, np.pi / 4 * (erf(1) + 1) ** 2), (-np.inf, -1, -np.inf, 1, np.pi / 4 * ((erf(1) + 1) * erfc(1))), (-np.inf, 1, -np.inf, -1, np.pi / 4 * ((erf(1) + 1) * erfc(1))), (0, np.inf, 0, np.inf, np.pi / 4), (1, np.inf, 0, np.inf, np.pi / 4 * erfc(1)), (0, np.inf, 1, np.inf, np.pi / 4 * erfc(1)), (1, np.inf, 1, np.inf, np.pi / 4 * erfc(1) ** 2), (-1, np.inf, 0, np.inf, np.pi / 4 * (erf(1) + 1)), (0, np.inf, -1, np.inf, np.pi / 4 * (erf(1) + 1)), (-1, np.inf, -1, np.inf, np.pi / 4 * (erf(1) + 1) ** 2), (-1, np.inf, 1, np.inf, np.pi / 4 * ((erf(1) + 1) * erfc(1))), (1, np.inf, -1, np.inf, np.pi / 4 * ((erf(1) + 1) * erfc(1))), (-np.inf, np.inf, -np.inf, np.inf, np.pi)])
def test_double_integral_improper(self, x_lower, x_upper, y_lower, y_upper, expected):

    def f(x, y):
        return np.exp(-x ** 2 - y ** 2)
    assert_quad(dblquad(f, x_lower, x_upper, y_lower, y_upper), expected, error_tolerance=3e-08)