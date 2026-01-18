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
def test_binom_exact(self):

    @np.vectorize
    def binom_int(n, k):
        n = int(n)
        k = int(k)
        num = 1
        den = 1
        for i in range(1, k + 1):
            num *= i + n - k
            den *= i
        return float(num / den)
    np.random.seed(1234)
    n = np.arange(1, 15)
    k = np.arange(0, 15)
    nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
    nk = nk[nk[:, 0] >= nk[:, 1]]
    assert_func_equal(cephes.binom, binom_int(nk[:, 0], nk[:, 1]), nk, atol=0, rtol=0)