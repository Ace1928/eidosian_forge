import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom,expected', [(point, point_z), (point_z, point_z), pytest.param(empty_point, empty_point_z, marks=empty_geom_mark), pytest.param(empty_point_z, empty_point_z, marks=empty_geom_mark), (line_string, line_string_z), (line_string_z, line_string_z), pytest.param(empty_line_string, empty_line_string_z, marks=empty_geom_mark), pytest.param(empty_line_string_z, empty_line_string_z, marks=empty_geom_mark), (polygon, polygon_z), (polygon_z, polygon_z), (polygon_with_hole, polygon_with_hole_z), (polygon_with_hole_z, polygon_with_hole_z), (multi_point, multi_point_z), (multi_point_z, multi_point_z), (multi_line_string, multi_line_string_z), (multi_line_string_z, multi_line_string_z), (multi_polygon, multi_polygon_z), (multi_polygon_z, multi_polygon_z), (geometry_collection_2, geometry_collection_z), (geometry_collection_z, geometry_collection_z)])
def test_force_3d(geom, expected):
    actual = shapely.force_3d(geom, z=4)
    assert shapely.get_coordinate_dimension(actual) == 3
    assert_geometries_equal(actual, expected)