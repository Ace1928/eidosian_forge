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
def test_from_wkt_exceptions():
    with pytest.raises(TypeError, match='Expected bytes or string, got int'):
        shapely.from_wkt(1)
    with pytest.raises(shapely.GEOSException, match='Expected word but encountered end of stream'):
        shapely.from_wkt('')
    with pytest.raises(shapely.GEOSException, match="Unknown type: 'NOT'"):
        shapely.from_wkt('NOT A WKT STRING')