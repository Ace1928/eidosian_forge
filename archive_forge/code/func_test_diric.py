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
def test_diric(self):
    n_odd = [1, 5, 25]
    x = np.array(2 * np.pi + 5e-05).astype(np.float32)
    assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=7)
    x = np.array(2 * np.pi + 1e-09).astype(np.float64)
    assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
    x = np.array(2 * np.pi + 1e-15).astype(np.float64)
    assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
    if hasattr(np, 'float128'):
        x = np.array(2 * np.pi + 1e-12).astype(np.float128)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=19)
    n_even = [2, 4, 24]
    x = np.array(2 * np.pi + 1e-09).astype(np.float64)
    assert_almost_equal(special.diric(x, n_even), -1.0, decimal=15)
    x = np.arange(0.2 * np.pi, 1.0 * np.pi, 0.2 * np.pi)
    octave_result = [0.872677996249965, 0.539344662916632, 0.127322003750035, -0.206011329583298]
    assert_almost_equal(special.diric(x, 3), octave_result, decimal=15)