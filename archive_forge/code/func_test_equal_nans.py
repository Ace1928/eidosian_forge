import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_equal_nans():
    s = GeoSeries([Point(0, 0), np.nan])
    assert_geoseries_equal(s, s.copy())
    assert_geoseries_equal(s, s.copy(), check_less_precise=True)