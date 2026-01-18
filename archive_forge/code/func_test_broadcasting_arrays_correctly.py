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
@pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-13})])
def test_broadcasting_arrays_correctly(self, is_exact, comp, kwargs):
    ans = asarray([[1, 15, 25, 10], [1, 7, 6, 1]])
    n = asarray([[5, 5, 5, 5], [4, 4, 4, 4]])
    k = asarray([1, 2, 3, 4])
    comp(stirling2(n, k, exact=is_exact), ans, **kwargs)
    n = asarray([[4], [4], [4], [4], [4]])
    k = asarray([0, 1, 2, 3, 4, 5])
    ans = asarray([[0, 1, 7, 6, 1, 0] for _ in range(5)])
    comp(stirling2(n, k, exact=False), ans, **kwargs)