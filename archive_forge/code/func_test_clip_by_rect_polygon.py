import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom, rect, expected', [(Polygon(((0, 0), (0, 30), (30, 30), (30, 0), (0, 0)), holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))]), (10, 10, 20, 20), GeometryCollection()), (Polygon(((0, 0), (0, 30), (30, 30), (30, 0), (0, 0)), holes=[((10, 10), (10, 20), (20, 20), (20, 10), (10, 10))]), (10, 10, 20, 20), GeometryCollection()), (Polygon(((1, 1), (1, 30), (30, 30), (30, 1), (1, 1)), holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))]), (0, 0, 40, 40), Polygon(((1, 1), (1, 30), (30, 30), (30, 1), (1, 1)), holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))])), (Polygon([(0, 0), (0, 30), (30, 30), (30, 0), (0, 0)], holes=[[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]]), (5, 5, 15, 15), Polygon([(5, 5), (5, 15), (10, 15), (10, 10), (15, 10), (15, 5), (5, 5)]))])
def test_clip_by_rect_polygon(geom, rect, expected):
    actual = shapely.clip_by_rect(geom, *rect)
    assert_geometries_equal(actual, expected)