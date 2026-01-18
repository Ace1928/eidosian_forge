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
def test_make_poly_3(self):
    c1 = BPoly._construct_from_derivatives(0, 1, [1, 2, 3], [4])
    assert_allclose(c1, [1.0, 5.0 / 3, 17.0 / 6, 4.0])
    c2 = BPoly._construct_from_derivatives(0, 1, [1], [4, 2, 3])
    assert_allclose(c2, [1.0, 19.0 / 6, 10.0 / 3, 4.0])
    c3 = BPoly._construct_from_derivatives(0, 1, [1, 2], [4, 3])
    assert_allclose(c3, [1.0, 5.0 / 3, 3.0, 4.0])