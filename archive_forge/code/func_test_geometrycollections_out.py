import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('indices,expected', [([0, 0], [geom_coll([point, line_string]), None, None, empty_point]), ([3, 3], [None, None, None, geom_coll([point, line_string])])])
def test_geometrycollections_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[3] = empty_point
    actual = shapely.geometrycollections([point, line_string], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out