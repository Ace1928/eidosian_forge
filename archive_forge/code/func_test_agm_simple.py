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
def test_agm_simple():
    rtol = 1e-13
    assert_allclose(1 / special.agm(1, np.sqrt(2)), 0.8346268416740732, rtol=rtol)
    agm13 = 1.863616783244897
    agm15 = 2.60400819053094
    agm35 = 3.936235503649555
    assert_allclose(special.agm([[1], [3]], [1, 3, 5]), [[1, agm13, agm15], [agm13, 3, agm35]], rtol=rtol)
    agm12 = 1.4567910310469068
    assert_allclose(special.agm(1, 2), agm12, rtol=rtol)
    assert_allclose(special.agm(2, 1), agm12, rtol=rtol)
    assert_allclose(special.agm(-1, -2), -agm12, rtol=rtol)
    assert_allclose(special.agm(24, 6), 13.458171481725614, rtol=rtol)
    assert_allclose(special.agm(13, 123456789.5), 11111458.498599306, rtol=rtol)
    assert_allclose(special.agm(1e+30, 1), 2.229223055945383e+28, rtol=rtol)
    assert_allclose(special.agm(1e-22, 1), 0.030182566420169886, rtol=rtol)
    assert_allclose(special.agm(1e+150, 1e+180), 2.229223055945383e+178, rtol=rtol)
    assert_allclose(special.agm(1e+180, 1e-150), 2.0634722510162677e+177, rtol=rtol)
    assert_allclose(special.agm(1e-150, 1e-170), 3.3112619670463756e-152, rtol=rtol)
    fi = np.finfo(1.0)
    assert_allclose(special.agm(fi.tiny, fi.max), 1.9892072050015473e+305, rtol=rtol)
    assert_allclose(special.agm(0.75 * fi.max, fi.max), 1.564904312298045e+308, rtol=rtol)
    assert_allclose(special.agm(fi.tiny, 3 * fi.tiny), 4.1466849866735005e-308, rtol=rtol)
    assert_equal(special.agm(0, 0), 0)
    assert_equal(special.agm(99, 0), 0)
    assert_equal(special.agm(-1, 10), np.nan)
    assert_equal(special.agm(0, np.inf), np.nan)
    assert_equal(special.agm(np.inf, 0), np.nan)
    assert_equal(special.agm(0, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, 0), np.nan)
    assert_equal(special.agm(np.inf, -np.inf), np.nan)
    assert_equal(special.agm(-np.inf, np.inf), np.nan)
    assert_equal(special.agm(1, np.nan), np.nan)
    assert_equal(special.agm(np.nan, -1), np.nan)
    assert_equal(special.agm(1, np.inf), np.inf)
    assert_equal(special.agm(np.inf, 1), np.inf)
    assert_equal(special.agm(-1, -np.inf), -np.inf)
    assert_equal(special.agm(-np.inf, -1), -np.inf)