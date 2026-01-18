import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
def test_geometric_transform05(self, order, dtype):
    data = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=dtype)
    expected = numpy.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]], dtype=dtype)
    if data.dtype.kind == 'c':
        data -= 1j * data
        expected -= 1j * expected

    def mapping(x):
        return (x[0], x[1] - 1)
    out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
    assert_array_almost_equal(out, expected)