import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_ignore_crs_mismatch():
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': s1.copy()}, crs='EPSG:4326')
    df2 = GeoDataFrame({'col1': [1, 2], 'geometry': s1}, crs='EPSG:31370')
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2)
    with warnings.catch_warnings(record=True) as record:
        assert_geodataframe_equal(df1, df2, check_crs=False)
    assert len(record) == 0