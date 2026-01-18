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
def test_ellipe(self):
    ele = special.ellipe(0.2)
    assert_almost_equal(ele, 1.489035058095853, 8)
    assert_equal(special.ellipe(0.0), pi / 2)
    assert_equal(special.ellipe(1.0), 1.0)
    assert_equal(special.ellipe(-np.inf), np.inf)
    assert_equal(special.ellipe(np.nan), np.nan)
    assert_equal(special.ellipe(2), np.nan)
    assert_allclose(special.ellipe(-10), 3.639138038417769)