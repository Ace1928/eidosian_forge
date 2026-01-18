import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.xfail
def test_no_overlapping_geometry(self):
    df_inner = sjoin(self.pointdf.iloc[17:], self.polydf, how='inner')
    df_left = sjoin(self.pointdf.iloc[17:], self.polydf, how='left')
    df_right = sjoin(self.pointdf.iloc[17:], self.polydf, how='right')
    expected_inner_df = pd.concat([self.pointdf.iloc[:0], pd.Series(name='index_right', dtype='int64'), self.polydf.drop('geometry', axis=1).iloc[:0]], axis=1)
    expected_inner = GeoDataFrame(expected_inner_df)
    expected_right_df = pd.concat([self.pointdf.drop('geometry', axis=1).iloc[:0], pd.concat([pd.Series(name='index_left', dtype='int64'), pd.Series(name='index_right', dtype='int64')], axis=1), self.polydf], axis=1)
    expected_right = GeoDataFrame(expected_right_df).set_index('index_right')
    expected_left_df = pd.concat([self.pointdf.iloc[17:], pd.Series(name='index_right', dtype='int64'), self.polydf.iloc[:0].drop('geometry', axis=1)], axis=1)
    expected_left = GeoDataFrame(expected_left_df)
    assert expected_inner.equals(df_inner)
    assert expected_right.equals(df_right)
    assert expected_left.equals(df_left)