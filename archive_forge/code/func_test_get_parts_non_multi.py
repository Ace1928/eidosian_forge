import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [point, line_string, polygon])
def test_get_parts_non_multi(geom):
    """Non-multipart geometries should be returned identical to inputs"""
    assert_geometries_equal(geom, shapely.get_parts(geom))