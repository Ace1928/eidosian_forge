import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_rings_return_index():
    geom = np.array([polygon, None, empty_polygon, polygon_with_hole])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        if g is None or shapely.is_empty(g):
            continue
        expected_parts.append(shapely.get_exterior_ring(g))
        expected_index.append(i)
        for j in range(0, shapely.get_num_interior_rings(g)):
            expected_parts.append(shapely.get_interior_ring(g, j))
            expected_index.append(i)
    parts, index = shapely.get_rings(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)