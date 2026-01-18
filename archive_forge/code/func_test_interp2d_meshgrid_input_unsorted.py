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
def test_interp2d_meshgrid_input_unsorted(self):
    np.random.seed(1234)
    x = linspace(0, 2, 16)
    y = linspace(0, pi, 21)
    z = sin(x[None, :] + y[:, None] / 2.0)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        ip1 = interp2d(x.copy(), y.copy(), z, kind='cubic')
        np.random.shuffle(x)
        z = sin(x[None, :] + y[:, None] / 2.0)
        ip2 = interp2d(x.copy(), y.copy(), z, kind='cubic')
        np.random.shuffle(x)
        np.random.shuffle(y)
        z = sin(x[None, :] + y[:, None] / 2.0)
        ip3 = interp2d(x, y, z, kind='cubic')
        x = linspace(0, 2, 31)
        y = linspace(0, pi, 30)
        assert_equal(ip1(x, y), ip2(x, y))
        assert_equal(ip1(x, y), ip3(x, y))