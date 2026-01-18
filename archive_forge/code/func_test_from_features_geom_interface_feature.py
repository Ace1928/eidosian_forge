import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_from_features_geom_interface_feature(self):

    class Placemark(object):

        def __init__(self, geom, val):
            self.__geo_interface__ = {'type': 'Feature', 'properties': {'a': val}, 'geometry': geom.__geo_interface__}
    p1 = Point(1, 1)
    f1 = Placemark(p1, 0)
    p2 = Point(3, 3)
    f2 = Placemark(p2, 0)
    df = GeoDataFrame.from_features([f1, f2])
    assert sorted(df.columns) == ['a', 'geometry']
    assert df.geometry.tolist() == [p1, p2]