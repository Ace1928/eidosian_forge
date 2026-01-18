import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def test_apply_axis1_secondary_geo_cols(df):
    geo_name = df.geometry.name

    def identity(x):
        return x
    assert_obj_no_active_geo_col(df[['geometry2']].apply(identity, axis=1), GeoDataFrame, geo_name)