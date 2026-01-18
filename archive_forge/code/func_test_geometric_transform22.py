import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform22(self, order):
    data = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], numpy.float64)

    def mapping1(x):
        return (x[0] / 2, x[1] / 2)

    def mapping2(x):
        return (x[0] * 2, x[1] * 2)
    out = ndimage.geometric_transform(data, mapping1, (6, 8), order=order)
    out = ndimage.geometric_transform(out, mapping2, (3, 4), order=order)
    assert_array_almost_equal(out, data)