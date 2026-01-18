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
def test_conversions(self):
    pp, bp = self._make_polynomials()
    pp1 = self.P.from_bernstein_basis(bp)
    assert_equal(pp1.__class__, self.P)
    bp1 = self.B.from_power_basis(pp)
    assert_equal(bp1.__class__, self.B)