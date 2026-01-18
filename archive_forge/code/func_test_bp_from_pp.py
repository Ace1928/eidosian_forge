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
def test_bp_from_pp(self):
    x = [0, 1, 3]
    c = [[3, 2], [1, 8], [4, 3]]
    pp = PPoly(c, x)
    bp = BPoly.from_power_basis(pp)
    pp1 = PPoly.from_bernstein_basis(bp)
    xp = [0.1, 1.4]
    assert_allclose(pp(xp), bp(xp))
    assert_allclose(pp(xp), pp1(xp))