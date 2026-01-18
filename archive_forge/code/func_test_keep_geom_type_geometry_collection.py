import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_keep_geom_type_geometry_collection():
    df1 = read_file(os.path.join(DATA, 'geom_type', 'df1.geojson'))
    df2 = read_file(os.path.join(DATA, 'geom_type', 'df2.geojson'))
    with pytest.warns(UserWarning, match='`keep_geom_type=True` in overlay'):
        intersection = overlay(df1, df2, keep_geom_type=None)
    assert len(intersection) == 1
    assert (intersection.geom_type == 'Polygon').all()
    intersection = overlay(df1, df2, keep_geom_type=True)
    assert len(intersection) == 1
    assert (intersection.geom_type == 'Polygon').all()
    intersection = overlay(df1, df2, keep_geom_type=False)
    assert len(intersection) == 1
    assert (intersection.geom_type == 'GeometryCollection').all()