import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_keep_geom_type_geomcoll_different_types():
    polys1 = [box(0, 1, 1, 3), box(10, 10, 12, 12)]
    polys2 = [Polygon([(1, 0), (3, 0), (3, 3), (1, 3), (1, 2), (2, 2), (2, 1), (1, 1)]), box(11, 11, 13, 13)]
    df1 = GeoDataFrame({'left': [0, 1], 'geometry': polys1})
    df2 = GeoDataFrame({'right': [0, 1], 'geometry': polys2})
    result1 = overlay(df1, df2, keep_geom_type=True)
    expected1 = GeoDataFrame({'left': [1], 'right': [1], 'geometry': [box(11, 11, 12, 12)]})
    assert_geodataframe_equal(result1, expected1)
    result2 = overlay(df1, df2, keep_geom_type=False)
    expected2 = GeoDataFrame({'left': [0, 1], 'right': [0, 1], 'geometry': [GeometryCollection([LineString([(1, 2), (1, 3)]), Point(1, 1)]), box(11, 11, 12, 12)]})
    assert_geodataframe_equal(result2, expected2)