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
def test_geometry_property(self):
    assert_geoseries_equal(self.df.geometry, self.df['geometry'], check_dtype=True, check_index_type=True)
    df = self.df.copy()
    new_geom = [Point(x, y) for x, y in zip(range(len(self.df)), range(len(self.df)))]
    df.geometry = new_geom
    new_geom = GeoSeries(new_geom, index=df.index, crs=df.crs)
    assert_geoseries_equal(df.geometry, new_geom)
    assert_geoseries_equal(df['geometry'], new_geom)
    gs = new_geom.to_crs(crs='epsg:3857')
    df.geometry = gs
    assert df.crs == 'epsg:3857'