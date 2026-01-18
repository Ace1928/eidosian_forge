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
def test_hyp2f1(self):
    values = [[0.5, 1, 1.5, 0.2 ** 2, 0.5 / 0.2 * log((1 + 0.2) / (1 - 0.2))], [0.5, 1, 1.5, -0.2 ** 2, 1.0 / 0.2 * arctan(0.2)], [1, 1, 2, 0.2, -1 / 0.2 * log(1 - 0.2)], [3, 3.5, 1.5, 0.2 ** 2, 0.5 / 0.2 / -5 * ((1 + 0.2) ** (-5) - (1 - 0.2) ** (-5))], [-3, 3, 0.5, sin(0.2) ** 2, cos(2 * 3 * 0.2)], [3, 4, 8, 1, special.gamma(8) * special.gamma(8 - 4 - 3) / special.gamma(8 - 3) / special.gamma(8 - 4)], [3, 2, 3 - 2 + 1, -1, 1.0 / 2 ** 3 * sqrt(pi) * special.gamma(1 + 3 - 2) / special.gamma(1 + 0.5 * 3 - 2) / special.gamma(0.5 + 0.5 * 3)], [5, 2, 5 - 2 + 1, -1, 1.0 / 2 ** 5 * sqrt(pi) * special.gamma(1 + 5 - 2) / special.gamma(1 + 0.5 * 5 - 2) / special.gamma(0.5 + 0.5 * 5)], [4, 0.5 + 4, 1.5 - 2 * 4, -1.0 / 3, (8.0 / 9) ** (-2 * 4) * special.gamma(4.0 / 3) * special.gamma(1.5 - 2 * 4) / special.gamma(3.0 / 2) / special.gamma(4.0 / 3 - 2 * 4)], [1.5, -0.5, 1.0, -10.0, 4.130009776527747], [-2, 3, 1, 0.95, 0.715], [2, -3, 1, 0.95, -0.007], [-6, 3, 1, 0.95, 8.10625e-05], [2, -5, 1, 0.95, -2.9375e-05], (10, -900, 10.5, 0.99, 1.9185370579660766e-24), (10, -900, -10.5, 0.99, 3.542792000403557e-18)]
    for i, (a, b, c, x, v) in enumerate(values):
        cv = special.hyp2f1(a, b, c, x)
        assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)