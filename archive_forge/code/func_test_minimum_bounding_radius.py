import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geometry, expected', [(Polygon([(0, 5), (5, 10), (10, 5), (5, 0), (0, 5)]), 5), (LineString([(1, 0), (1, 10)]), 5), (MultiPoint([(2, 2), (4, 2)]), 1), (Point(2, 2), 0), (GeometryCollection(), 0)])
def test_minimum_bounding_radius(geometry, expected):
    actual = shapely.minimum_bounding_radius(geometry)
    assert actual == pytest.approx(expected, abs=1e-12)