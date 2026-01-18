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
def test_big_numbers(self):
    ans = asarray([48063331393110, 48004081105038305])
    n = [25, 30]
    k = [17, 4]
    assert array_equal(stirling2(n, k, exact=True), ans)
    ans = asarray([2801934359500572414253157841233849412, 14245032222277144547280648984426251])
    n = [42, 43]
    k = [17, 23]
    assert array_equal(stirling2(n, k, exact=True), ans)