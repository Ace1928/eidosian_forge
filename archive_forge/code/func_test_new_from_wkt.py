import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', all_types)
def test_new_from_wkt(geom):
    actual = shapely.from_wkt(str(geom))
    assert_geometries_equal(actual, geom)