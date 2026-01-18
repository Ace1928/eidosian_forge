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
def test_airy(self):
    x = special.airy(0.99)
    assert_array_almost_equal(x, array([0.13689066, -0.16050153, 1.19815925, 0.92046818]), 8)
    x = special.airy(0.41)
    assert_array_almost_equal(x, array([0.25238916, -0.23480512, 0.80686202, 0.51053919]), 8)
    x = special.airy(-0.36)
    assert_array_almost_equal(x, array([0.44508477, -0.23186773, 0.44939534, 0.48105354]), 8)