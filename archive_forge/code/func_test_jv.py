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
def test_jv(self):
    values = [[0, 0.1, 0.99750156206604], [2.0 / 3, 1e-08, 3.239028506761532e-06], [2.0 / 3, 1e-10, 1.503423854873779e-07], [3.1, 1e-10, 1.711956265409013e-33], [2.0 / 3, 4.0, -0.2325440850267039]]
    for i, (v, x, y) in enumerate(values):
        yc = special.jv(v, x)
        assert_almost_equal(yc, y, 8, err_msg='test #%d' % i)