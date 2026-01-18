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
def test_elliprj(self):
    assert_allclose(elliprj(1, 1, 1, 1), 1)
    assert elliprj(1, 1, inf, 1) == 0.0
    assert isnan(elliprj(1, 0, 0, 0))
    assert isnan(elliprj(-1, 1, 1, 1))
    assert elliprj(1, 1, 1, inf) == 0.0
    args = array([[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, -1.0 + 1j], [1j, -1j, 0.0, 2.0], [-1.0 + 1j, -1.0 - 1j, 1.0, 2.0], [1j, -1j, 0.0, 1.0 - 1j], [-1.0 + 1j, -1.0 - 1j, 1.0, -3.0 + 1j], [2.0, 3.0, 4.0, -0.5], [2.0, 3.0, 4.0, -5.0]])
    expected_results = array([0.77688623778582, 0.14297579667157, 0.13613945827771 - 0.38207561624427j, 1.6490011662711, 0.9414835884122, 1.8260115229009 + 1.2290661908643j, -0.61127970812028 - 1.0684038390007j, 0.24723819703052, -0.12711230042964])
    for i, arr in enumerate(args):
        assert_allclose(elliprj(*arr), expected_results[i])