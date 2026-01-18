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
def test_interp2d_bounds(self):
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 2, 7)
    z = x[None, :] ** 2 + y[:, None]
    ix = np.linspace(-1, 3, 31)
    iy = np.linspace(-1, 3, 33)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        b = interp2d(x, y, z, bounds_error=True)
        assert_raises(ValueError, b, ix, iy)
        b = interp2d(x, y, z, fill_value=np.nan)
        iz = b(ix, iy)
        mx = (ix < 0) | (ix > 1)
        my = (iy < 0) | (iy > 2)
        assert_(np.isnan(iz[my, :]).all())
        assert_(np.isnan(iz[:, mx]).all())
        assert_(np.isfinite(iz[~my, :][:, ~mx]).all())