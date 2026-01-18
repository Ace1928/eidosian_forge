import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
def test_distance_duplicated():
    a = Point(1, 2)
    b = LineString([(0, 0), (0, 0), (1, 1)])
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        actual = shapely.distance(a, b)
    assert actual == 1.0