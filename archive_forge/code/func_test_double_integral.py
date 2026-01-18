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
def test_double_integral(self):

    def simpfunc(y, x):
        return x + y
    a, b = (1.0, 2.0)
    assert_quad(dblquad(simpfunc, a, b, lambda x: x, lambda x: 2 * x), 5 / 6.0 * (b ** 3.0 - a ** 3.0))