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
def test_hyperu(self):
    val1 = special.hyperu(1, 0.1, 100)
    assert_almost_equal(val1, 0.0098153, 7)
    a, b = ([0.3, 0.6, 1.2, -2.7], [1.5, 3.2, -0.4, -3.2])
    a, b = (asarray(a), asarray(b))
    z = 0.5
    hypu = special.hyperu(a, b, z)
    hprl = pi / sin(pi * b) * (special.hyp1f1(a, b, z) / (special.gamma(1 + a - b) * special.gamma(b)) - z ** (1 - b) * special.hyp1f1(1 + a - b, 2 - b, z) / (special.gamma(a) * special.gamma(2 - b)))
    assert_array_almost_equal(hypu, hprl, 12)