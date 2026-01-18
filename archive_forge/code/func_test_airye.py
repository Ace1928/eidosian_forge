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
def test_airye(self):
    a = special.airye(0.01)
    b = special.airy(0.01)
    b1 = [None] * 4
    for n in range(2):
        b1[n] = b[n] * exp(2.0 / 3.0 * 0.01 * sqrt(0.01))
    for n in range(2, 4):
        b1[n] = b[n] * exp(-abs(real(2.0 / 3.0 * 0.01 * sqrt(0.01))))
    assert_array_almost_equal(a, b1, 6)