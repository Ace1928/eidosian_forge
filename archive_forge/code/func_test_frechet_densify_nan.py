import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
def test_frechet_densify_nan():
    actual = shapely.frechet_distance(line_string, line_string, densify=np.nan)
    assert np.isnan(actual)