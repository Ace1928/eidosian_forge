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
def test_interp2d_meshgrid_input(self):
    x = linspace(0, 2, 16)
    y = linspace(0, pi, 21)
    z = sin(x[None, :] + y[:, None] / 2.0)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        II = interp2d(x, y, z)
        assert_almost_equal(II(1.0, 2.0), sin(2.0), decimal=2)