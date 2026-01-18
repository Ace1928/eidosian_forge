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
def test_from_wkb_ignore_on_invalid():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        shapely.from_wkt('', on_invalid='ignore')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        shapely.from_wkt('NOT A WKT STRING', on_invalid='ignore')