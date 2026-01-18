import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def test_expandim_in_groupby_aggregate_multiple_funcs():
    s = GeoSeries.from_xy([0, 1, 2], [0, 1, 3])

    def union(s):
        return s.unary_union

    def total_area(s):
        return s.area.sum()
    grouped = s.groupby([0, 1, 0])
    agg = grouped.agg([total_area, union])
    assert_obj_no_active_geo_col(agg, GeoDataFrame, geo_colname=None)
    result = grouped.agg([union, total_area])
    assert_obj_no_active_geo_col(result, GeoDataFrame, geo_colname=None)
    assert_object(grouped.agg([total_area, total_area]), pd.DataFrame)
    assert_object(grouped.agg([total_area]), pd.DataFrame)