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
def test_km1(self):
    """Test identity:
        K(m) = R_F(0, 1-m, 1)
        But with the ellipkm1 function
        """
    tiny = finfo(double).tiny
    m1 = tiny * 2.0 ** arange(0.0, -np.log2(tiny))
    assert_allclose(ellipkm1(m1), elliprf(0.0, m1, 1.0))