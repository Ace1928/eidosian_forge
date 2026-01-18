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
def test_dataframe_not_manipulated(self):
    df = pd.DataFrame({'A': range(len(self.df)), 'latitude': self.df.geometry.centroid.y, 'longitude': self.df.geometry.centroid.x}, index=self.df.index)
    df_copy = df.copy()
    gf = GeoDataFrame(df, geometry=points_from_xy(df['longitude'], df['latitude']), crs=self.df.crs)
    assert type(df) == pd.DataFrame
    assert 'geometry' not in df
    assert_frame_equal(df, df_copy)
    assert isinstance(gf, GeoDataFrame)
    assert hasattr(gf, 'geometry')
    gf.loc[0, 'A'] = 7
    assert_frame_equal(df, df_copy)
    gf['A'] = 3
    assert_frame_equal(df, df_copy)