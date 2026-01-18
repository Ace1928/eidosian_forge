import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_crs_axis_order__always_xy():
    df = GeoDataFrame(geometry=[Point(-1683723, 6689139)], crs='epsg:26918')
    lonlat = df.to_crs('epsg:4326')
    test_lonlat = GeoDataFrame(geometry=[Point(-110.1399901, 55.1350011)], crs='epsg:4326')
    assert_geodataframe_equal(lonlat, test_lonlat, check_less_precise=True)