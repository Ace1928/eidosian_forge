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
def test_multi_shape(self):
    c = np.random.rand(6, 2, 1, 2, 3)
    x = np.array([0, 0.5, 1])
    p = BPoly(c, x)
    assert_equal(p.x.shape, x.shape)
    assert_equal(p.c.shape, c.shape)
    assert_equal(p(0.3).shape, c.shape[2:])
    assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])
    dp = p.derivative()
    assert_equal(dp.c.shape, (5, 2, 1, 2, 3))