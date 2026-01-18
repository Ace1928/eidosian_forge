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
def test_deriv_inplace(self):
    np.random.seed(1234)
    m, k = (5, 8)
    x = np.sort(np.random.random(m))
    c = np.random.random((k, m - 1))
    for cc in [c.copy(), c * (1.0 + 2j)]:
        bp = BPoly(cc, x)
        xp = np.linspace(x[0], x[-1], 21)
        for i in range(k):
            assert_allclose(bp(xp, i), bp.derivative(i)(xp))