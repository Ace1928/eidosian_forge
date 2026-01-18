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
def test_deriv_3d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 6, 7, 8, 9)
    x = np.linspace(0, 1, 7 + 1)
    y = np.linspace(0, 1, 8 + 1) ** 2
    z = np.linspace(0, 1, 9 + 1) ** 3
    p = NdPPoly(c, (x, y, z))
    p1 = PPoly(c.transpose(0, 3, 1, 2, 4, 5), x)
    dp = p.derivative(nu=[2])
    dp1 = p1.derivative(2)
    assert_allclose(dp.c, dp1.c.transpose(0, 2, 3, 1, 4, 5))
    p1 = PPoly(c.transpose(1, 4, 0, 2, 3, 5), y)
    dp = p.antiderivative(nu=[0, 1, 0])
    dp1 = p1.antiderivative(1)
    assert_allclose(dp.c, dp1.c.transpose(2, 0, 3, 4, 1, 5))
    p1 = PPoly(c.transpose(2, 5, 0, 1, 3, 4), z)
    dp = p.derivative(nu=[0, 0, 3])
    dp1 = p1.derivative(3)
    assert_allclose(dp.c, dp1.c.transpose(2, 3, 0, 4, 5, 1))