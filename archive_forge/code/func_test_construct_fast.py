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
def test_construct_fast(self):
    np.random.seed(1234)
    c = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
    x = np.array([0, 0.5, 1])
    p = PPoly.construct_fast(c, x)
    assert_allclose(p(0.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
    assert_allclose(p(0.7), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)