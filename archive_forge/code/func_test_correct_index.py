import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_correct_index(dfs):
    df1, df2 = dfs
    polys3 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df3 = GeoDataFrame({'geometry': polys3, 'col3': [1, 2, 3]})
    i1 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
    i2 = Polygon([(3, 3), (3, 5), (5, 5), (5, 3), (3, 3)])
    expected = GeoDataFrame([[1, 1, i1], [3, 2, i2]], columns=['col3', 'col2', 'geometry'])
    result = overlay(df3, df2, keep_geom_type=True)
    assert_geodataframe_equal(result, expected)