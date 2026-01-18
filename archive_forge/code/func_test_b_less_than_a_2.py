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
def test_b_less_than_a_2(self):

    def f(x, s):
        return np.exp(-x ** 2 / 2 / s) / np.sqrt(2.0 * s)
    val_1, err_1 = quad(f, -np.inf, np.inf, args=(2,))
    val_2, err_2 = quad(f, np.inf, -np.inf, args=(2,))
    assert_allclose(val_1, -val_2, atol=max(err_1, err_2))