import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('order', ['C', 'F'])
def test_get_coords_index_multidim(order):
    geometry = np.array([[point, line_string], [empty, empty]], order=order)
    expected = [0, 1, 1, 1]
    _, actual = get_coordinates(geometry, return_index=True)
    assert_equal(actual, expected)