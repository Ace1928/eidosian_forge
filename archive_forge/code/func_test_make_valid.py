import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geom,expected', [(point, point), (Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]), MultiLineString([((1, 1), (1, 2)), ((0, 0), (1, 1))])), (Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]), MultiPolygon([Polygon([(1, 1), (2, 2), (2, 0), (1, 1)]), Polygon([(0, 0), (0, 2), (1, 1), (0, 0)])])), (empty, empty), ([empty], [empty])])
def test_make_valid(geom, expected):
    actual = shapely.make_valid(geom)
    assert actual is not expected
    assert shapely.normalize(actual) == expected