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
def test_betaln(self):
    assert_equal(special.betaln(1, 1), 0.0)
    assert_allclose(special.betaln(-100.3, 1e-200), special.gammaln(1e-200))
    assert_allclose(special.betaln(0.0342, 170), 3.1811881124242447, rtol=1e-14, atol=0)
    betln = special.betaln(2, 4)
    bet = log(abs(special.beta(2, 4)))
    assert_allclose(betln, bet, rtol=1e-13)