import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [point, line_string, linear_ring, multi_point, multi_line_string, multi_polygon, geometry_collection])
def test_get_exterior_ring_non_polygon(geom):
    actual = shapely.get_exterior_ring(geom)
    assert shapely.is_missing(actual).all()