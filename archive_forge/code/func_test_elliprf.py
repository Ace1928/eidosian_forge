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
def test_elliprf(self):
    assert_allclose(elliprf(1, 1, 1), 1)
    assert_allclose(elliprf(0, 1, 2), 1.3110287771460598)
    assert elliprf(1, inf, 1) == 0.0
    assert np.isinf(elliprf(0, 1, 0))
    assert isnan(elliprf(1, 1, -1))
    assert elliprf(complex(inf), 0, 1) == 0.0
    assert isnan(elliprf(1, 1, complex(-inf, 1)))
    args = array([[1.0, 2.0, 0.0], [1j, -1j, 0.0], [0.5, 1.0, 0.0], [-1.0 + 1j, 1j, 0.0], [2.0, 3.0, 4.0], [1j, -1j, 2.0], [-1.0 + 1j, 1j, 1.0 - 1j]])
    expected_results = array([1.3110287771461, 1.8540746773014, 1.8540746773014, 0.79612586584234 - 1.2138566698365j, 0.58408284167715, 1.0441445654064, 0.93912050218619 - 0.53296252018635j])
    for i, arr in enumerate(args):
        assert_allclose(elliprf(*arr), expected_results[i])