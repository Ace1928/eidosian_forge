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
def test_antiderivative_continuity(self):
    c = np.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
    x = np.array([0, 0.5, 1])
    p = PPoly(c, x)
    ip = p.antiderivative()
    assert_allclose(ip(0.5 - 1e-09), ip(0.5 + 1e-09), rtol=1e-08)
    p2 = ip.derivative()
    assert_allclose(p2.c, p.c)