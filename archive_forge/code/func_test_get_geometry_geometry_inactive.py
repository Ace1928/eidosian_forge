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
def test_get_geometry_geometry_inactive(self):
    df = self.df.assign(geom2=self.df.geometry).set_geometry('geom2')
    df = df.loc[:, ['BoroName', 'geometry']]
    assert df._geometry_column_name == 'geom2'
    msg_geo_col_missing = 'is not present. '
    with pytest.raises(AttributeError, match=msg_geo_col_missing):
        df.geometry