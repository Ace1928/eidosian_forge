import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform06(self, order):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])

    def mapping(x):
        return (x[0], x[1] - 1)
    out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
    assert_array_almost_equal(out, [[0, 4, 1, 3], [0, 7, 6, 8], [0, 3, 5, 3]])