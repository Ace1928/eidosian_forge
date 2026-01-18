import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_empty_join_due_to_max_distance_how_right(self):
    left = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
    right = geopandas.GeoDataFrame({'geometry': [Point(2, 2)]})
    joined = sjoin_nearest(left, right, how='right', max_distance=1, distance_col='distances')
    expected = right.copy()
    expected['index_left'] = [np.nan]
    expected['distances'] = [np.nan]
    expected = expected[['index_left', 'geometry', 'distances']]
    assert_geodataframe_equal(joined, expected)