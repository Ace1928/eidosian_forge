import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def test_expanddim_in_unstack():
    s = GeoSeries.from_xy([0, 1, 2], [0, 1, 3], index=pd.MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a')]))
    unstack = s.unstack()
    expected_geo_name = None
    assert_obj_no_active_geo_col(unstack, GeoDataFrame, geo_colname=expected_geo_name)
    s.name = 'geometry'
    unstack = s.unstack()
    assert_obj_no_active_geo_col(unstack, GeoDataFrame, expected_geo_name)