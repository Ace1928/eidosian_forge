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
def test_jnyn_zeros(self):
    jnz = special.jnyn_zeros(1, 5)
    assert_array_almost_equal(jnz, (array([3.83171, 7.01559, 10.17347, 13.32369, 16.47063]), array([1.84118, 5.33144, 8.53632, 11.706, 14.86359]), array([2.19714, 5.42968, 8.59601, 11.74915, 14.89744]), array([3.68302, 6.9415, 10.1234, 13.28576, 16.44006])), 5)