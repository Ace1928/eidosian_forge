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
def test_antider_ppoly(self):
    np.random.seed(1234)
    x = np.sort(np.random.random(11))
    c = np.random.random((4, 10, 2, 3))
    bp = BPoly(c, x)
    pp = PPoly.from_bernstein_basis(bp)
    xx = np.linspace(x[0], x[-1], 10)
    assert_allclose(bp.antiderivative(2)(xx), pp.antiderivative(2)(xx), atol=1e-12, rtol=1e-12)