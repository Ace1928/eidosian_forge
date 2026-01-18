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
def test_cosine_weighted_infinite(self):

    def myfunc(x, a):
        return exp(x * a)
    a = 2.5
    ome = 2.3
    assert_quad(quad(myfunc, -np.inf, 0, args=a, weight='cos', wvar=ome), a / (a ** 2 + ome ** 2))