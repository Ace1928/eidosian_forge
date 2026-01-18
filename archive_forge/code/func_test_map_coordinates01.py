import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
def test_map_coordinates01(self, order, dtype):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    expected = numpy.array([[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])
    if data.dtype.kind == 'c':
        data = data - 1j * data
        expected = expected - 1j * expected
    idx = numpy.indices(data.shape)
    idx -= 1
    out = ndimage.map_coordinates(data, idx, order=order)
    assert_array_almost_equal(out, expected)