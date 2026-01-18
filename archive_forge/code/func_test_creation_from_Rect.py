import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_creation_from_Rect(self):
    for nux, nuy in self.orders:
        lut_der = self.lut_rect.partial_derivative(nux, nuy)
        a = lut_der(0.5, 1.5, grid=False)
        b = self.lut_rect(0.5, 1.5, dx=nux, dy=nuy, grid=False)
        assert_equal(a, b)