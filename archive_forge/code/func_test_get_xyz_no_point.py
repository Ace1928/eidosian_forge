import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('func', [shapely.get_x, shapely.get_y, pytest.param(shapely.get_z, marks=pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7'))])
@pytest.mark.parametrize('geom', all_types[1:])
def test_get_xyz_no_point(func, geom):
    assert np.isnan(func(geom))