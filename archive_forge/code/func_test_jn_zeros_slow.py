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
def test_jn_zeros_slow(self):
    jn0 = special.jn_zeros(0, 300)
    assert_allclose(jn0[260 - 1], 816.0288449506887, rtol=1e-13)
    assert_allclose(jn0[280 - 1], 878.8606870712442, rtol=1e-13)
    assert_allclose(jn0[300 - 1], 941.6925306531796, rtol=1e-13)
    jn10 = special.jn_zeros(10, 300)
    assert_allclose(jn10[260 - 1], 831.6766851430563, rtol=1e-13)
    assert_allclose(jn10[280 - 1], 894.5127509537132, rtol=1e-13)
    assert_allclose(jn10[300 - 1], 957.3482637086654, rtol=1e-13)
    jn3010 = special.jn_zeros(3010, 5)
    assert_allclose(jn3010, array([3036.86590780927, 3057.06598526482, 3073.66360690272, 3088.37736494778, 3101.86438139042]), rtol=1e-08)