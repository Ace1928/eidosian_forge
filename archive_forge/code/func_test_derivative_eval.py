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
def test_derivative_eval(self):
    np.random.seed(1234)
    x = np.sort(np.r_[0, np.random.rand(11), 1])
    y = np.random.rand(len(x))
    spl = splrep(x, y, s=0)
    pp = PPoly.from_spline(spl)
    xi = np.linspace(0, 1, 200)
    for dx in range(0, 3):
        assert_allclose(pp(xi, dx), splev(xi, spl, dx))