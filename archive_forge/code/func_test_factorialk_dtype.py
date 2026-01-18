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
@pytest.mark.parametrize('k', range(1, 12))
def test_factorialk_dtype(self, k):
    if k in _FACTORIALK_LIMITS_64BITS.keys():
        n = np.array([_FACTORIALK_LIMITS_32BITS[k]])
        assert_equal(special.factorialk(n, k).dtype, np_long)
        assert_equal(special.factorialk(n + 1, k).dtype, np.int64)
        assert special.factorialk(n + 1, k) > np.iinfo(np.int32).max
        n = np.array([_FACTORIALK_LIMITS_64BITS[k]])
        assert_equal(special.factorialk(n, k).dtype, np.int64)
        assert_equal(special.factorialk(n + 1, k).dtype, object)
        assert special.factorialk(n + 1, k) > np.iinfo(np.int64).max
    else:
        assert_equal(special.factorialk(np.array([1]), k).dtype, object)