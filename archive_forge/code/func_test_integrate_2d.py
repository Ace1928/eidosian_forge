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
def test_integrate_2d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 16, 17)
    x = np.linspace(0, 1, 16 + 1) ** 1
    y = np.linspace(0, 1, 17 + 1) ** 2
    c = c.transpose(0, 2, 1, 3)
    cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
    _ppoly.fix_continuity(cx, x, 2)
    c = cx.reshape(c.shape)
    c = c.transpose(0, 2, 1, 3)
    c = c.transpose(1, 3, 0, 2)
    cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
    _ppoly.fix_continuity(cx, y, 2)
    c = cx.reshape(c.shape)
    c = c.transpose(2, 0, 3, 1).copy()
    p = NdPPoly(c, (x, y))
    for ranges in [[(0, 1), (0, 1)], [(0, 0.5), (0, 1)], [(0, 1), (0, 0.5)], [(0.3, 0.7), (0.6, 0.2)]]:
        ig = p.integrate(ranges)
        ig2, err2 = nquad(lambda x, y: p((x, y)), ranges, opts=[dict(epsrel=1e-05, epsabs=1e-05)] * 2)
        assert_allclose(ig, ig2, rtol=1e-05, atol=1e-05, err_msg=repr(ranges))