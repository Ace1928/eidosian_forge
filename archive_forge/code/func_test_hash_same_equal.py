import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', all_types + (shapely.points(np.nan, np.nan), empty_point))
def test_hash_same_equal(geom):
    assert hash(geom) == hash(shapely.transform(geom, lambda x: x))