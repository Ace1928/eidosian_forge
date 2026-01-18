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
@pytest.mark.xfail(reason='The real part of expm1(z) bad at these points')
def test_expm1_complex_hard(self):
    y = np.array([0.1, 0.2, 0.3, 5, 11, 20])
    x = -np.log(np.cos(y))
    z = x + 1j * y
    expected = np.array([-5.5507901846769623e-17 + 0.10033467208545054j, 2.4289354732893695e-18 + 0.20271003550867248j, 4.523550026258577e-17 + 0.3093362496096232j, 7.8234305217489e-17 - 3.3805150062465863j, -1.3685191953697676e-16 - 225.95084645419513j, 8.717562048129105e-17 + 2.237160944224742j])
    found = cephes.expm1(z)
    assert_array_almost_equal_nulp(found.imag, expected.imag, 3)
    assert_array_almost_equal_nulp(found.real, expected.real, 20)