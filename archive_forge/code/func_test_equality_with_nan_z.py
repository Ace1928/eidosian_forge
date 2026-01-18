import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
@pytest.mark.parametrize('left, right', cases2)
def test_equality_with_nan_z(left, right):
    if shapely.geos_version < (3, 12, 0):
        assert left == right
        assert not left != right
    else:
        assert left != right