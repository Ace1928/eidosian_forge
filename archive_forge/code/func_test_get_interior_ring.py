import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_interior_ring():
    actual = shapely.get_interior_ring(polygon_with_hole, [0, -1, 1, -2])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()