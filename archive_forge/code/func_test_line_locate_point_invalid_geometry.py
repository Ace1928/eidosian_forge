import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('normalized', [False, True])
def test_line_locate_point_invalid_geometry(normalized):
    with pytest.raises(shapely.GEOSException):
        shapely.line_locate_point(line_string, line_string, normalized=normalized)
    with pytest.raises(shapely.GEOSException):
        shapely.line_locate_point(polygon, point, normalized=normalized)