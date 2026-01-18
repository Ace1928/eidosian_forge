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
def test_huber():
    assert_equal(special.huber(-1, 1.5), np.inf)
    assert_allclose(special.huber(2, 1.5), 0.5 * np.square(1.5))
    assert_allclose(special.huber(2, 2.5), 2 * (2.5 - 0.5 * 2))

    def xfunc(delta, r):
        if delta < 0:
            return np.inf
        elif np.abs(r) < delta:
            return 0.5 * np.square(r)
        else:
            return delta * (np.abs(r) - 0.5 * delta)
    z = np.random.randn(10, 2)
    w = np.vectorize(xfunc, otypes=[np.float64])(z[:, 0], z[:, 1])
    assert_func_equal(special.huber, w, z, rtol=1e-13, atol=1e-13)