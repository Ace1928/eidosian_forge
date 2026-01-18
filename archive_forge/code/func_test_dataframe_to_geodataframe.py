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
def test_dataframe_to_geodataframe(self):
    df = pd.DataFrame({'A': range(len(self.df)), 'location': np.array(self.df.geometry)}, index=self.df.index)
    gf = df.set_geometry('location', crs=self.df.crs)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(gf, GeoDataFrame)
    assert_geoseries_equal(gf.geometry, self.df.geometry)
    assert gf.geometry.name == 'location'
    assert 'geometry' not in gf
    gf2 = df.set_geometry('location', crs=self.df.crs, drop=True)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(gf2, GeoDataFrame)
    assert gf2.geometry.name == 'geometry'
    assert 'geometry' in gf2
    assert 'location' not in gf2
    assert 'location' in df
    df.loc[0, 'A'] = 100
    assert gf.loc[0, 'A'] == 0
    assert gf2.loc[0, 'A'] == 0
    with pytest.raises(ValueError):
        df.set_geometry('location', inplace=True)