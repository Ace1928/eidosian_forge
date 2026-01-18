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
def test_ellipkinc(self):
    elkinc = special.ellipkinc(pi / 2, 0.2)
    elk = special.ellipk(0.2)
    assert_almost_equal(elkinc, elk, 15)
    alpha = 20 * pi / 180
    phi = 45 * pi / 180
    m = sin(alpha) ** 2
    elkinc = special.ellipkinc(phi, m)
    assert_almost_equal(elkinc, 0.79398143, 8)
    assert_equal(special.ellipkinc(pi / 2, 0.0), pi / 2)
    assert_equal(special.ellipkinc(pi / 2, 1.0), np.inf)
    assert_equal(special.ellipkinc(pi / 2, -np.inf), 0.0)
    assert_equal(special.ellipkinc(pi / 2, np.nan), np.nan)
    assert_equal(special.ellipkinc(pi / 2, 2), np.nan)
    assert_equal(special.ellipkinc(0, 0.5), 0.0)
    assert_equal(special.ellipkinc(np.inf, 0.5), np.inf)
    assert_equal(special.ellipkinc(-np.inf, 0.5), -np.inf)
    assert_equal(special.ellipkinc(np.inf, np.inf), np.nan)
    assert_equal(special.ellipkinc(np.inf, -np.inf), np.nan)
    assert_equal(special.ellipkinc(-np.inf, -np.inf), np.nan)
    assert_equal(special.ellipkinc(-np.inf, np.inf), np.nan)
    assert_equal(special.ellipkinc(np.nan, 0.5), np.nan)
    assert_equal(special.ellipkinc(np.nan, np.nan), np.nan)
    assert_allclose(special.ellipkinc(0.3897411203531872, 1), 0.4, rtol=1e-14)
    assert_allclose(special.ellipkinc(1.5707, -10), 0.7908428466172495)