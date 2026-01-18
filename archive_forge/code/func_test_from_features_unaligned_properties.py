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
def test_from_features_unaligned_properties(self):
    p1 = Point(1, 1)
    f1 = {'type': 'Feature', 'properties': {'a': 0}, 'geometry': p1.__geo_interface__}
    p2 = Point(2, 2)
    f2 = {'type': 'Feature', 'properties': {'b': 1}, 'geometry': p2.__geo_interface__}
    p3 = Point(3, 3)
    f3 = {'type': 'Feature', 'properties': None, 'geometry': p3.__geo_interface__}
    df = GeoDataFrame.from_features([f1, f2, f3])
    result = df[['a', 'b']]
    expected = pd.DataFrame.from_dict([{'a': 0, 'b': np.nan}, {'a': np.nan, 'b': 1}, {'a': np.nan, 'b': np.nan}])
    assert_frame_equal(expected, result)