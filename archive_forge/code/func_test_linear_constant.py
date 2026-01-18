import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_linear_constant(self):
    assert_almost_equal(self.lut.get_residual(), 0.0)
    assert_array_almost_equal(self.lut([1, 1.5, 2], [1, 1.5]), [[3, 3], [3, 3], [3, 3]])