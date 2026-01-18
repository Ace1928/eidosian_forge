import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
def test_kl_div():

    def xfunc(x, y):
        if x < 0 or y < 0 or (y == 0 and x != 0):
            return np.inf
        elif np.isposinf(x) or np.isposinf(y):
            return np.inf
        elif x == 0:
            return y
        else:
            return special.xlogy(x, x / y) - x + y
    values = (0, 0.5, 1.0)
    signs = [-1, 1]
    arr = []
    for sgna, va, sgnb, vb in itertools.product(signs, values, signs, values):
        arr.append((sgna * va, sgnb * vb))
    z = np.array(arr, dtype=float)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:, 0], z[:, 1])
    assert_func_equal(special.kl_div, w, z, rtol=1e-13, atol=1e-13)