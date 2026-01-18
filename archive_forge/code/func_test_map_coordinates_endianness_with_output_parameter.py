import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_map_coordinates_endianness_with_output_parameter(self):
    data = numpy.array([[1, 2], [7, 6]])
    expected = numpy.array([[0, 0], [0, 1]])
    idx = numpy.indices(data.shape)
    idx -= 1
    for out in [data.dtype, data.dtype.newbyteorder(), numpy.empty_like(expected), numpy.empty_like(expected).astype(expected.dtype.newbyteorder())]:
        returned = ndimage.map_coordinates(data, idx, output=out)
        result = out if returned is None else returned
        assert_array_almost_equal(result, expected)