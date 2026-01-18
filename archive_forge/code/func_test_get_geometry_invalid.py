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
def test_get_geometry_invalid(self):
    df = GeoDataFrame()
    df['geom'] = self.df.geometry
    msg_geo_col_none = 'active geometry column to use has not been set. '
    with pytest.raises(AttributeError, match=msg_geo_col_none):
        df.geometry
    col_subset_drop_geometry = ['BoroCode', 'BoroName', 'geom2']
    df2 = self.df.copy().assign(geom2=self.df.geometry)[col_subset_drop_geometry]
    with pytest.raises(AttributeError, match='is not present.'):
        df2.geometry
    msg_other_geo_cols_present = 'There are columns with geometry data type'
    msg_no_other_geo_cols = 'There are no existing columns with geometry data type'
    with pytest.raises(AttributeError, match=msg_other_geo_cols_present):
        df2.geometry
    with pytest.raises(AttributeError, match=msg_no_other_geo_cols):
        GeoDataFrame().geometry