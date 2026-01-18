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
def test_deriv_clpmn(self):
    zvals = [0.5 + 0.5j, -0.5 + 0.5j, -0.5 - 0.5j, 0.5 - 0.5j, 1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    m = 2
    n = 3
    for type in [2, 3]:
        for z in zvals:
            for h in [0.001, 0.001j]:
                approx_derivative = (special.clpmn(m, n, z + 0.5 * h, type)[0] - special.clpmn(m, n, z - 0.5 * h, type)[0]) / h
                assert_allclose(special.clpmn(m, n, z, type)[1], approx_derivative, rtol=0.0001)