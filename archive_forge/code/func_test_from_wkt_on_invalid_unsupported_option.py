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
def test_from_wkt_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match='not a valid option'):
        shapely.from_wkt(b'\x01\x01\x00\x00\x00\x00', on_invalid='unsupported_option')