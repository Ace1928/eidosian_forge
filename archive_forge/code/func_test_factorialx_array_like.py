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
@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('level', range(1, 5))
def test_factorialx_array_like(self, level, exact):

    def _nest_me(x, k=1):
        if k == 0:
            return x
        else:
            return _nest_me([x], k - 1)
    n = _nest_me([5], k=level - 1)
    exp_nucleus = {1: 120, 2: 15, 3: 10}
    assert_func = assert_array_equal if exact else assert_allclose
    assert_func(special.factorial(n, exact=exact), np.array(exp_nucleus[1], ndmin=level))
    assert_func(special.factorial2(n, exact=exact), np.array(exp_nucleus[2], ndmin=level))
    assert_func(special.factorialk(n, 3, exact=True), np.array(exp_nucleus[3], ndmin=level))