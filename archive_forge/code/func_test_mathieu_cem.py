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
def test_mathieu_cem(self):
    assert_equal(cephes.mathieu_cem(1, 0, 0), (1.0, 0.0))

    @np.vectorize
    def ce_smallq(m, q, z):
        z *= np.pi / 180
        if m == 0:
            return 2 ** (-0.5) * (1 - 0.5 * q * cos(2 * z))
        elif m == 1:
            return cos(z) - q / 8 * cos(3 * z)
        elif m == 2:
            return cos(2 * z) - q * (cos(4 * z) / 12 - 1 / 4)
        else:
            return cos(m * z) - q * (cos((m + 2) * z) / (4 * (m + 1)) - cos((m - 2) * z) / (4 * (m - 1)))
    m = np.arange(0, 100)
    q = np.r_[0, np.logspace(-30, -9, 10)]
    assert_allclose(cephes.mathieu_cem(m[:, None], q[None, :], 0.123)[0], ce_smallq(m[:, None], q[None, :], 0.123), rtol=1e-14, atol=0)