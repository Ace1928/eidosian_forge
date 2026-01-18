import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [point, line_string, linear_ring, multi_point, multi_line_string, geometry_collection])
def test_area_non_polygon(geom):
    assert shapely.area(geom) == 0.0