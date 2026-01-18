import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_coordinate_dimension():
    actual = shapely.get_coordinate_dimension([point, point_z, None]).tolist()
    assert actual == [2, 3, -1]