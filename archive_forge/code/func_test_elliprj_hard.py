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
@pytest.mark.xfail(reason='Insufficient accuracy on 32-bit')
def test_elliprj_hard(self):
    assert_allclose(elliprj(6.483625725195452e-08, 1.1649136528196886e-27, 36767340167168.0, 0.493704617023468), 8.634269206442419e-06, rtol=5e-15, atol=1e-20)
    assert_allclose(elliprj(14.375105857849121, 9.993988969725365e-11, 1.72844262269944e-26, 5.898871222598245e-06), 829774.1424801627, rtol=5e-15, atol=1e-20)