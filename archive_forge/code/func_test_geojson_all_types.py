import json
import pickle
import struct
import warnings
import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LineString, Point, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty_point, empty_point_z, point, point_z
@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason='GEOS < 3.10.1')
@pytest.mark.parametrize('geom', all_types)
def test_geojson_all_types(geom):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip('Linearrings are not preserved in GeoJSON')
    geojson = shapely.to_geojson(geom)
    actual = shapely.from_geojson(geojson)
    assert_geometries_equal(actual, geom)