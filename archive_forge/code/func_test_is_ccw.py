from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
@pytest.mark.parametrize('geom, expected', [(LinearRing([(0, 0), (0, 1), (1, 1), (0, 0)]), False), (LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]), True), (LineString([(0, 0), (0, 1), (1, 1), (0, 0)]), False), (LineString([(0, 0), (1, 1), (0, 1), (0, 0)]), True), (LineString([(0, 0), (1, 1), (0, 1)]), False), (LineString([(0, 0), (0, 1), (1, 1)]), False), (point, False), (polygon, False), (geometry_collection, False), (None, False)])
def test_is_ccw(geom, expected):
    assert shapely.is_ccw(geom) == expected