import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_keep_geom_type_geometry_collection2():
    polys1 = [box(0, 0, 1, 1), box(1, 1, 3, 3).union(box(1, 3, 5, 5))]
    polys2 = [box(0, 0, 1, 1), box(3, 1, 4, 2).union(box(4, 1, 5, 4))]
    df1 = GeoDataFrame({'left': [0, 1], 'geometry': polys1})
    df2 = GeoDataFrame({'right': [0, 1], 'geometry': polys2})
    result1 = overlay(df1, df2, keep_geom_type=True)
    expected1 = GeoDataFrame({'left': [0, 1], 'right': [0, 1], 'geometry': [box(0, 0, 1, 1), box(4, 3, 5, 4)]})
    assert_geodataframe_equal(result1, expected1)
    result1 = overlay(df1, df2, keep_geom_type=False)
    expected1 = GeoDataFrame({'left': [0, 1, 1], 'right': [0, 0, 1], 'geometry': [box(0, 0, 1, 1), Point(1, 1), GeometryCollection([box(4, 3, 5, 4), LineString([(3, 1), (3, 2)])])]})
    assert_geodataframe_equal(result1, expected1)