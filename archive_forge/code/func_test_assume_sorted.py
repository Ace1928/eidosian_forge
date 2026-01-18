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
def test_assume_sorted(self):
    interp10 = interp1d(self.x10, self.y10)
    interp10_unsorted = interp1d(self.x10[::-1], self.y10[::-1])
    assert_array_almost_equal(interp10_unsorted(self.x10), self.y10)
    assert_array_almost_equal(interp10_unsorted(1.2), np.array([1.2]))
    assert_array_almost_equal(interp10_unsorted([2.4, 5.6, 6.0]), interp10([2.4, 5.6, 6.0]))
    interp10_assume_kw = interp1d(self.x10[::-1], self.y10[::-1], assume_sorted=False)
    assert_array_almost_equal(interp10_assume_kw(self.x10), self.y10)
    interp10_assume_kw2 = interp1d(self.x10[::-1], self.y10[::-1], assume_sorted=True)
    assert_raises(ValueError, interp10_assume_kw2, self.x10)
    interp10_y_2d = interp1d(self.x10, self.y210)
    interp10_y_2d_unsorted = interp1d(self.x10[::-1], self.y210[:, ::-1])
    assert_array_almost_equal(interp10_y_2d(self.x10), interp10_y_2d_unsorted(self.x10))