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
def test_double_integral2(self):

    def func(x0, x1, t0, t1):
        return x0 + x1 + t0 + t1

    def g(x):
        return x

    def h(x):
        return 2 * x
    args = (1, 2)
    assert_quad(dblquad(func, 1, 2, g, h, args=args), 35.0 / 6 + 9 * 0.5)