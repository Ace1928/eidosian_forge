import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_map_coordinates02(self, order):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    idx = numpy.indices(data.shape, numpy.float64)
    idx -= 0.5
    out1 = ndimage.shift(data, 0.5, order=order)
    out2 = ndimage.map_coordinates(data, idx, order=order)
    assert_array_almost_equal(out1, out2)