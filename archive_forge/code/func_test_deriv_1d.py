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
def test_deriv_1d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5)
    x = np.linspace(0, 1, 5 + 1)
    p = NdPPoly(c, (x,))
    dp = p.derivative(nu=[1])
    p1 = PPoly(c, x)
    dp1 = p1.derivative()
    assert_allclose(dp.c, dp1.c)
    dp = p.antiderivative(nu=[2])
    p1 = PPoly(c, x)
    dp1 = p1.antiderivative(2)
    assert_allclose(dp.c, dp1.c)