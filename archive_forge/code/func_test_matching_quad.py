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
def test_matching_quad(self):

    def func(x):
        return x ** 2 + 1
    res, reserr = quad(func, 0, 4)
    res2, reserr2 = nquad(func, ranges=[[0, 4]])
    assert_almost_equal(res, res2)
    assert_almost_equal(reserr, reserr2)