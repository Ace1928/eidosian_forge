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
def test_cauchypv_weight(self):

    def myfunc(x, a):
        return 2.0 ** (-a) / ((x - 1) ** 2 + 4.0 ** (-a))
    a = 0.4
    tabledValue = (2.0 ** (-0.4) * log(1.5) - 2.0 ** (-1.4) * log((4.0 ** (-a) + 16) / (4.0 ** (-a) + 1)) - arctan(2.0 ** (a + 2)) - arctan(2.0 ** a)) / (4.0 ** (-a) + 1)
    assert_quad(quad(myfunc, 0, 5, args=0.4, weight='cauchy', wvar=2.0), tabledValue, error_tolerance=1.9e-08)