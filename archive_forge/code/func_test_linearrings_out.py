import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('indices,expected', [([0, 0, 0, 0, 0], [linear_ring, None, None, empty_point]), ([1, 1, 1, 1, 1], [None, linear_ring, None, empty_point]), ([3, 3, 3, 3, 3], [None, None, None, linear_ring])])
def test_linearrings_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[3] = empty_point
    actual = shapely.linearrings([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out