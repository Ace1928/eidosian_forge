import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_dataframe_setitem_without_geometry_column(self):
    arr = from_shapely(self.geoms)
    df = GeoDataFrame({'col1': [1, 2], 'geometry': arr}, crs=4326)
    with pytest.warns(UserWarning):
        df['geometry'] = 1
    df['geometry'] = self.geoms
    assert df.crs is None