import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_keep_geom_type_geometry_collection_difference():
    polys1 = [box(0, 0, 1, 1), box(1, 1, 2, 2)]
    polys2 = [box(0, 0, 1, 1), box(1, 1, 2, 3).union(box(2, 2, 3, 2.0))]
    df1 = GeoDataFrame({'left': [0, 1], 'geometry': polys1})
    df2 = GeoDataFrame({'right': [0, 1], 'geometry': polys2})
    result1 = overlay(df2, df1, keep_geom_type=True, how='difference')
    expected1 = GeoDataFrame({'right': [1], 'geometry': [box(1, 2, 2, 3)]})
    assert_geodataframe_equal(result1, expected1)