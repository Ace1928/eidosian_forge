import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_distance_transform_cdt02(self, dtype):
    data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
    out, ft = ndimage.distance_transform_cdt(data, 'chessboard', return_indices=True)
    bf = ndimage.distance_transform_bf(data, 'chessboard')
    assert_array_almost_equal(bf, out)
    expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 1, 1, 1, 2, 2, 2], [3, 3, 2, 2, 1, 2, 2, 3, 3], [4, 4, 3, 2, 2, 2, 3, 4, 4], [5, 5, 4, 6, 7, 6, 4, 5, 5], [6, 6, 6, 6, 7, 7, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8, 8, 8, 8]], [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 2, 3, 4, 6, 7, 8], [0, 1, 1, 2, 2, 6, 6, 7, 8], [0, 1, 1, 1, 2, 6, 7, 7, 8], [0, 1, 1, 2, 6, 6, 7, 7, 8], [0, 1, 2, 2, 5, 6, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
    assert_array_almost_equal(ft, expected)