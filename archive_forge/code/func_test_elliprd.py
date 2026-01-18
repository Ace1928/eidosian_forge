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
def test_elliprd(self):
    assert_allclose(elliprd(1, 1, 1), 1)
    assert_allclose(elliprd(0, 2, 1) / 3.0, 0.5990701173677961)
    assert elliprd(1, 1, inf) == 0.0
    assert np.isinf(elliprd(1, 1, 0))
    assert np.isinf(elliprd(1, 1, complex(0, 0)))
    assert np.isinf(elliprd(0, 1, complex(0, 0)))
    assert isnan(elliprd(1, 1, -np.finfo(np.float64).tiny / 2.0))
    assert isnan(elliprd(1, 1, complex(-1, 0)))
    args = array([[0.0, 2.0, 1.0], [2.0, 3.0, 4.0], [1j, -1j, 2.0], [0.0, 1j, -1j], [0.0, -1.0 + 1j, 1j], [-2.0 - 1j, -1j, -1.0 + 1j]])
    expected_results = array([1.7972103521034, 0.16510527294261, 0.6593385415422, 1.270819627191 + 2.7811120159521j, -1.8577235439239 - 0.96193450888839j, 1.8249027393704 - 1.2218475784827j])
    for i, arr in enumerate(args):
        assert_allclose(elliprd(*arr), expected_results[i])