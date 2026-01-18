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
def test_ellipkinc_singular(self):
    xlog = np.logspace(-300, -17, 25)
    xlin = np.linspace(1e-17, 0.1, 25)
    xlin2 = np.linspace(0.1, pi / 2, 25, endpoint=False)
    assert_allclose(special.ellipkinc(xlog, 1), np.arcsinh(np.tan(xlog)), rtol=100000000000000.0)
    assert_allclose(special.ellipkinc(xlin, 1), np.arcsinh(np.tan(xlin)), rtol=100000000000000.0)
    assert_allclose(special.ellipkinc(xlin2, 1), np.arcsinh(np.tan(xlin2)), rtol=100000000000000.0)
    assert_equal(special.ellipkinc(np.pi / 2, 1), np.inf)
    assert_allclose(special.ellipkinc(-xlog, 1), np.arcsinh(np.tan(-xlog)), rtol=100000000000000.0)
    assert_allclose(special.ellipkinc(-xlin, 1), np.arcsinh(np.tan(-xlin)), rtol=100000000000000.0)
    assert_allclose(special.ellipkinc(-xlin2, 1), np.arcsinh(np.tan(-xlin2)), rtol=100000000000000.0)
    assert_equal(special.ellipkinc(-np.pi / 2, 1), np.inf)