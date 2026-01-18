import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason='GEOS >= 3.10')
def set_precision_pointwise_pre_310():
    with pytest.warns(UserWarning):
        actual = shapely.set_precision(LineString([(0, 0), (0.1, 0.1)]), 1.0, mode='pointwise')
    assert_geometries_equal(shapely.force_2d(actual), LineString())