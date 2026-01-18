import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('crs', [None, 'EPSG:4326'])
def test_apply_no_geometry_result(df, crs):
    if crs:
        df = df.set_crs(crs)
    result = df.apply(lambda col: col.astype(str), axis=0)
    assert type(result) is pd.DataFrame
    expected = df.astype(str)
    assert_frame_equal(result, expected)
    result = df.apply(lambda col: col.astype(str), axis=1)
    assert type(result) is pd.DataFrame
    assert_frame_equal(result, expected)