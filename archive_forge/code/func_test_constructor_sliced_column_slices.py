import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def test_constructor_sliced_column_slices(df2):
    geo_idx = df2.columns.get_loc('geometry')
    sub = df2.head(1)
    assert type(sub.iloc[:, geo_idx]) == GeoSeries
    assert type(sub.iloc[[0], geo_idx]) == GeoSeries
    sub = df2.head(2)
    assert type(sub.iloc[:, geo_idx]) == GeoSeries
    assert type(sub.iloc[[0, 1], geo_idx]) == GeoSeries
    assert type(df2.iloc[0, :]) == pd.Series