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
@pytest.mark.parametrize('a, b, x, p', [(2.5, 3.0, 0.25, 0.833251953125), (7.5, 13.25, 0.375, 0.43298734645560366), (0.125, 7.5, 0.425, 0.0006688257851314237), (0.125, 18.0, 1e-06, 0.7298235914509633), (0.125, 18.0, 0.996, 7.274587553838015e-46), (0.125, 24.0, 0.75, 3.70853404816862e-17), (16.0, 0.75, 0.99999999975, 5.440875927741863e-07), (0.4211959643503401, 16939.046996018118, 0.000815296167195521, 1e-07)])
def test_betaincc_betainccinv(self, a, b, x, p):
    p1 = special.betaincc(a, b, x)
    assert_allclose(p1, p, rtol=5e-15)
    x1 = special.betainccinv(a, b, p)
    assert_allclose(x1, x, rtol=8e-15)