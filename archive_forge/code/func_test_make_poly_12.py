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
def test_make_poly_12(self):
    np.random.seed(12345)
    ya = np.r_[0, np.random.random(5)]
    yb = np.r_[0, np.random.random(5)]
    c = BPoly._construct_from_derivatives(0, 1, ya, yb)
    pp = BPoly(c[:, None], [0, 1])
    for j in range(6):
        assert_allclose([pp(0.0), pp(1.0)], [ya[j], yb[j]])
        pp = pp.derivative()