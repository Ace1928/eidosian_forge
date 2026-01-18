import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform10(self, order):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])

    def mapping(x):
        return (x[0] - 1, x[1] - 1)
    if order > 1:
        filtered = ndimage.spline_filter(data, order=order)
    else:
        filtered = data
    out = ndimage.geometric_transform(filtered, mapping, data.shape, order=order, prefilter=False)
    assert_array_almost_equal(out, [[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])