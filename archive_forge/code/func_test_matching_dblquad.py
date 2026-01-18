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
def test_matching_dblquad(self):

    def func2d(x0, x1):
        return x0 ** 2 + x1 ** 3 - x0 * x1 + 1
    res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
    res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
    assert_almost_equal(res, res2)
    assert_almost_equal(reserr, reserr2)