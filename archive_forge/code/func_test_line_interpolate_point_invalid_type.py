import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('geom', [empty_point, point, polygon, multi_point, multi_polygon, shapely.geometrycollections([point]), shapely.geometrycollections([polygon]), shapely.geometrycollections([multi_line_string]), shapely.geometrycollections([multi_point]), shapely.geometrycollections([multi_polygon])])
def test_line_interpolate_point_invalid_type(geom, normalized):
    with pytest.raises(TypeError):
        assert shapely.line_interpolate_point(geom, 0.2, normalized=normalized)