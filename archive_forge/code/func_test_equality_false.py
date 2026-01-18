import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
@pytest.mark.parametrize('left, right', [(LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 2)])), (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1 + 1e-12)])), (LineString([(0, 0), (1, 1)]), LineString([(1, 1), (0, 0)])), (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1), (1, 1)])), (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0.5, 0.5), (1, 1)])), (MultiLineString([[(1, 1), (2, 2)], [(2, 2), (3, 3)]]), MultiLineString([[(2, 2), (3, 3)], [(1, 1), (2, 2)]]))])
def test_equality_false(left, right):
    assert left != right