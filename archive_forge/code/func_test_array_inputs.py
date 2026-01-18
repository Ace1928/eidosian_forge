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
@pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
def test_array_inputs(self, is_exact, comp, kwargs):
    ans = [self.table[10][3], self.table[10][4]]
    comp(stirling2(asarray([10, 10]), asarray([3, 4]), exact=is_exact), ans)
    comp(stirling2([10, 10], asarray([3, 4]), exact=is_exact), ans)
    comp(stirling2(asarray([10, 10]), [3, 4], exact=is_exact), ans)