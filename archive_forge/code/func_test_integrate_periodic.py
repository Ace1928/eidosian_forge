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
def test_integrate_periodic(self):
    x = np.array([1, 2, 4])
    c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
    P = BPoly.from_power_basis(PPoly(c, x), extrapolate='periodic')
    I = P.antiderivative()
    period_int = I(4) - I(1)
    assert_allclose(P.integrate(1, 4), period_int)
    assert_allclose(P.integrate(-10, -7), period_int)
    assert_allclose(P.integrate(-10, -4), 2 * period_int)
    assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
    assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
    assert_allclose(P.integrate(3.5 + 12, 5 + 12), I(2) - I(1) + I(4) - I(3.5))
    assert_allclose(P.integrate(3.5, 5 + 12), I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)
    assert_allclose(P.integrate(0, -1), I(2) - I(3))
    assert_allclose(P.integrate(-9, -10), I(2) - I(3))
    assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)