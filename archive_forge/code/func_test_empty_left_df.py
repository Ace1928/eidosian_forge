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
@pytest.mark.parametrize('how', ['inner', 'left'])
@pytest.mark.parametrize('distance_col', (None, 'distance'))
def test_empty_left_df(self, how, distance_col: str):
    right = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
    left = geopandas.GeoDataFrame({'geometry': []})
    joined = sjoin_nearest(left, right, how=how, distance_col=distance_col)
    assert joined.empty
    if distance_col is not None:
        assert distance_col in joined