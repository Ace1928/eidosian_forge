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
def test_euler(self):
    eu0 = special.euler(0)
    eu1 = special.euler(1)
    eu2 = special.euler(2)
    assert_allclose(eu0, [1], rtol=1e-15)
    assert_allclose(eu1, [1, 0], rtol=1e-15)
    assert_allclose(eu2, [1, 0, -1], rtol=1e-15)
    eu24 = special.euler(24)
    mathworld = [1, 1, 5, 61, 1385, 50521, 2702765, 199360981, 19391512145, 2404879675441, 370371188237525, 69348874393137901, 15514534163557086905]
    correct = zeros((25,), 'd')
    for k in range(0, 13):
        if k % 2:
            correct[2 * k] = -float(mathworld[k])
        else:
            correct[2 * k] = float(mathworld[k])
    with np.errstate(all='ignore'):
        err = nan_to_num((eu24 - correct) / correct)
        errmax = max(err)
    assert_almost_equal(errmax, 0.0, 14)