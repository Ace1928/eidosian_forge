from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def test_extrapolate_attr(self):
    x = [0, 2]
    c = [[3], [1], [4]]
    bp = BPoly(c, x)
    for extrapolate in (True, False, None):
        bp = BPoly(c, x, extrapolate=extrapolate)
        bp_d = bp.derivative()
        if extrapolate is False:
            assert_(np.isnan(bp([-0.1, 2.1])).all())
            assert_(np.isnan(bp_d([-0.1, 2.1])).all())
        else:
            assert_(not np.isnan(bp([-0.1, 2.1])).any())
            assert_(not np.isnan(bp_d([-0.1, 2.1])).any())