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
@pytest.mark.parametrize('levels', range(1, 5))
@pytest.mark.parametrize('exact', [True, False])
def test_factorialx_array_shape(self, levels, exact):

    def _nest_me(x, k=1):
        """
            Double x and nest it k times

            For example:
            >>> _nest_me([3, 4], 2)
            [[[3, 4], [3, 4]], [[3, 4], [3, 4]]]
            """
        if k == 0:
            return x
        else:
            return _nest_me([x, x], k - 1)

    def _check(res, nucleus):
        exp = np.array(_nest_me(nucleus, k=levels), dtype=object)
        assert_allclose(res.astype(np.float64), exp.astype(np.float64))
    n = np.array(_nest_me([5, 25], k=levels))
    exp_nucleus = {1: [120, math.factorial(25)], 2: [15, special.factorial2(25, exact=True)], 3: [10, special.factorialk(25, 3)]}
    _check(special.factorial(n, exact=exact), exp_nucleus[1])
    _check(special.factorial2(n, exact=exact), exp_nucleus[2])
    _check(special.factorialk(n, 3, exact=True), exp_nucleus[3])