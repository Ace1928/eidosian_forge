import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_type_id():
    actual = shapely.get_type_id(all_types).tolist()
    assert actual == [0, 1, 2, 3, 4, 5, 6, 7, 7]