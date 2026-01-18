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
def test_interp2d_linear(self):
    a = np.zeros([5, 5])
    a[2, 2] = 1.0
    x = y = np.arange(5)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        b = interp2d(x, y, a, 'linear')
        assert_almost_equal(b(2.0, 1.5), np.array([0.5]), decimal=2)
        assert_almost_equal(b(2.0, 2.5), np.array([0.5]), decimal=2)