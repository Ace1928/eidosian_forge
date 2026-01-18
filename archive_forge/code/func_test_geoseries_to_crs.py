import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_geoseries_to_crs(self):
    s = GeoSeries(self.geoms, crs=27700)
    s = s.to_crs(4326)
    assert s.crs == self.wgs
    assert s.values.crs == self.wgs
    df = GeoDataFrame(geometry=s)
    assert df.crs == self.wgs
    df = df.to_crs(27700)
    assert df.crs == self.osgb
    assert df.geometry.crs == self.osgb
    assert df.geometry.values.crs == self.osgb
    arr = from_shapely(self.geoms, crs=4326)
    df['col1'] = arr
    df = df.to_crs(3857)
    assert df.col1.crs == self.wgs
    assert df.col1.values.crs == self.wgs