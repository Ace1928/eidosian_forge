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
def test_assign_cols_using_index(self):
    nybb_filename = geopandas.datasets.get_path('nybb')
    df = read_file(nybb_filename)
    other_df = pd.DataFrame({'foo': range(5), 'bar': range(5)})
    expected = pd.concat([df, other_df], axis=1)
    df[other_df.columns] = other_df
    assert_geodataframe_equal(df, expected)