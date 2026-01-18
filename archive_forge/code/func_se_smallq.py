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
@np.vectorize
def se_smallq(m, q, z):
    z *= np.pi / 180
    if m == 1:
        return sin(z) - q / 8 * sin(3 * z)
    elif m == 2:
        return sin(2 * z) - q * sin(4 * z) / 12
    else:
        return sin(m * z) - q * (sin((m + 2) * z) / (4 * (m + 1)) - sin((m - 2) * z) / (4 * (m - 1)))