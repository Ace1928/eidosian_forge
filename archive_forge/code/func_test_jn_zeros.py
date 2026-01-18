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
def test_jn_zeros(self):
    jn0 = special.jn_zeros(0, 5)
    jn1 = special.jn_zeros(1, 5)
    assert_array_almost_equal(jn0, array([2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391, 14.9309177086]), 4)
    assert_array_almost_equal(jn1, array([3.83171, 7.01559, 10.17347, 13.32369, 16.47063]), 4)
    jn102 = special.jn_zeros(102, 5)
    assert_allclose(jn102, array([110.8917493599204, 117.83464175788309, 123.70194191713507, 129.02417238949093, 134.00114761868423]), rtol=1e-13)
    jn301 = special.jn_zeros(301, 5)
    assert_allclose(jn301, array([313.5909786669883, 323.2154977609629, 331.2233873865675, 338.39676338872084, 345.03284233056064]), rtol=1e-13)