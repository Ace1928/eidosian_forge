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
def test_from_feature_collection(self):
    data = {'name': ['a', 'b', 'c'], 'lat': [45, 46, 47.5], 'lon': [-120, -121.2, -122.9]}
    df = pd.DataFrame(data)
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    expected = gdf[['geometry', 'name', 'lat', 'lon']]
    res = GeoDataFrame.from_features(gdf.__geo_interface__)
    assert_frame_equal(res, expected)
    res = GeoDataFrame.from_features(gdf.__geo_interface__['features'])
    assert_frame_equal(res, expected)
    res = GeoDataFrame.from_features(gdf)
    assert_frame_equal(res, expected)