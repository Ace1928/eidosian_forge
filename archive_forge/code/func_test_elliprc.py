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
def test_elliprc(self):
    assert_allclose(elliprc(1, 1), 1)
    assert elliprc(1, inf) == 0.0
    assert isnan(elliprc(1, 0))
    assert elliprc(1, complex(1, inf)) == 0.0
    args = array([[0.0, 0.25], [2.25, 2.0], [0.0, 1j], [-1j, 1j], [0.25, -2.0], [1j, -1.0]])
    expected_results = array([np.pi, np.log(2.0), 1.1107207345396 * (1.0 - 1j), 1.2260849569072 - 0.34471136988768j, np.log(2.0) / 3.0, 0.77778596920447 + 0.19832484993429j])
    for i, arr in enumerate(args):
        assert_allclose(elliprc(*arr), expected_results[i])