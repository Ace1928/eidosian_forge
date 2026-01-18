import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
@pytest.mark.parametrize('geom', [Point(), LineString(), Polygon(), MultiPoint(), MultiLineString(), MultiPolygon(), GeometryCollection(), LinearRing()])
def test_empty_geometry_bounds(geom):
    """The bounds of an empty geometry is a tuple of NaNs"""
    assert len(geom.bounds) == 4
    assert all((math.isnan(v) for v in geom.bounds))