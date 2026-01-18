import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('normalized', [False, True])
def test_line_locate_point_empty(normalized):
    assert np.isnan(shapely.line_locate_point(line_string, empty_point, normalized=normalized))
    assert np.isnan(shapely.line_locate_point(empty_line_string, point, normalized=normalized))