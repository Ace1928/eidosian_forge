import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geometry', all_types)
def test_minimum_bounding_circle_all_types(geometry):
    actual = shapely.minimum_bounding_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)
    actual = shapely.minimum_bounding_circle(None)
    assert actual is None