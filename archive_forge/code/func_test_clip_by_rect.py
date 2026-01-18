import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom,expected', [(Point(0, 0), GeometryCollection()), (Point(15, 15), Point(15, 15)), (Point(15, 10), GeometryCollection()), (LineString([(0, 0), (-5, 5)]), GeometryCollection()), (LineString([(15, 15), (16, 15)]), LineString([(15, 15), (16, 15)])), (LineString([(10, 15), (10, 10), (15, 10)]), GeometryCollection()), (LineString([(10, 5), (25, 20)]), LineString([(15, 10), (20, 15)]))])
def test_clip_by_rect(geom, expected):
    actual = shapely.clip_by_rect(geom, 10, 10, 20, 20)
    assert_geometries_equal(actual, expected)