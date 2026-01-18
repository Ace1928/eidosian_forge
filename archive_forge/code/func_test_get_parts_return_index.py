import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_parts_return_index():
    geom = np.array([multi_point, point, multi_polygon])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        for j in range(0, shapely.get_num_geometries(g)):
            expected_parts.append(shapely.get_geometry(g, j))
            expected_index.append(i)
    parts, index = shapely.get_parts(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)