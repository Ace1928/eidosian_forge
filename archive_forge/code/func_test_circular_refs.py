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
@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_circular_refs(self):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    with assert_deallocated(interp1d, x, y) as interp:
        interp([0.1, 0.2])
        del interp