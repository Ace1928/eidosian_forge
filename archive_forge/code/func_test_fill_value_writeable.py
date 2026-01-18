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
def test_fill_value_writeable(self):
    interp = interp1d(self.x10, self.y10, fill_value=123.0)
    assert_equal(interp.fill_value, 123.0)
    interp.fill_value = 321.0
    assert_equal(interp.fill_value, 321.0)