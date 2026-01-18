import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason='GEOS < 3.6')
def test_minimum_clearance_nonexistent():
    actual = shapely.minimum_clearance([point, empty])
    assert np.isinf(actual).all()