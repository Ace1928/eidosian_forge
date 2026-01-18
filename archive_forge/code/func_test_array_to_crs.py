import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_array_to_crs(self):
    arr = from_shapely(self.geoms, crs=27700)
    arr = arr.to_crs(4326)
    assert arr.crs == self.wgs