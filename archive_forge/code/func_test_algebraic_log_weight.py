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
def test_algebraic_log_weight(self):

    def myfunc(x, a):
        return 1 / (1 + x + 2 ** (-a))
    a = 1.5
    assert_quad(quad(myfunc, -1, 1, args=a, weight='alg', wvar=(-0.5, -0.5)), pi / sqrt((1 + 2 ** (-a)) ** 2 - 1))