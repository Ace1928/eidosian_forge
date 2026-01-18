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
def test_extend_descending(self):
    np.random.seed(0)
    order = 3
    x = np.sort(np.random.uniform(0, 10, 20))
    c = np.random.rand(order + 1, x.shape[0] - 1, 2, 3)
    for cls in (PPoly, BPoly):
        p = cls(c, x)
        p1 = cls(c[:, :9], x[:10])
        p1.extend(c[:, 9:], x[10:])
        p2 = cls(c[:, 10:], x[10:])
        p2.extend(c[:, :10], x[:10])
        assert_array_equal(p1.c, p.c)
        assert_array_equal(p1.x, p.x)
        assert_array_equal(p2.c, p.c)
        assert_array_equal(p2.x, p.x)