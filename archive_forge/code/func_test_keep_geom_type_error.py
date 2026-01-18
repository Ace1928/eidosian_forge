import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_keep_geom_type_error():
    gcol = GeoSeries(GeometryCollection([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), LineString([(3, 3), (5, 3), (5, 5), (3, 5)])]))
    dfcol = GeoDataFrame({'col1': [2], 'geometry': gcol})
    polys1 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': polys1})
    with pytest.raises(TypeError):
        overlay(dfcol, df1, keep_geom_type=True)