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
@pytest.mark.parametrize('geom', all_types)
@pytest.mark.parametrize('use_hex', [False, True])
@pytest.mark.parametrize('byte_order', [0, 1])
def test_from_wkb_all_types(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip('Linearrings are not preserved in WKB')
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)