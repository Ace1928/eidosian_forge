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
def test_comb(self):
    assert_array_almost_equal(special.comb([10, 10], [3, 4]), [120.0, 210.0])
    assert_almost_equal(special.comb(10, 3), 120.0)
    assert_equal(special.comb(10, 3, exact=True), 120)
    assert_equal(special.comb(10, 3, exact=True, repetition=True), 220)
    assert_allclose([special.comb(20, k, exact=True) for k in range(21)], special.comb(20, list(range(21))), atol=1e-15)
    ii = np.iinfo(int).max + 1
    assert_equal(special.comb(ii, ii - 1, exact=True), ii)
    expected = 100891344545564193334812497256
    assert special.comb(100, 50, exact=True) == expected