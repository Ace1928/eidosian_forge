import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_num_geometries():
    actual = shapely.get_num_geometries(all_types + (None,)).tolist()
    assert actual == [1, 1, 1, 1, 2, 1, 2, 2, 0, 0]