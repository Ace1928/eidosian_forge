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
def test_hyp0f1(self):
    assert_allclose(special.hyp0f1(2.5, 0.5), 1.21482702689997, rtol=1e-12)
    assert_allclose(special.hyp0f1(2.5, 0), 1.0, rtol=1e-15)
    x = special.hyp0f1(3.0, [-1.5, -1, 0, 1, 1.5])
    expected = np.array([0.58493659229143, 0.70566805723127, 1.0, 1.37789689539747, 1.6037368528848])
    assert_allclose(x, expected, rtol=1e-12)
    x = special.hyp0f1(3.0, np.array([-1.5, -1, 0, 1, 1.5]) + 0j)
    assert_allclose(x, expected.astype(complex), rtol=1e-12)
    x1 = [0.5, 1.5, 2.5]
    x2 = [0, 1, 0.5]
    x = special.hyp0f1(x1, x2)
    expected = [1.0, 1.8134302039235093, 1.21482702689997]
    assert_allclose(x, expected, rtol=1e-12)
    x = special.hyp0f1(np.vstack([x1] * 2), x2)
    assert_allclose(x, np.vstack([expected] * 2), rtol=1e-12)
    assert_raises(ValueError, special.hyp0f1, np.vstack([x1] * 3), [0, 1])