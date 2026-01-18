import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_almost_equal_but_not_equal():
    s_origin = GeoSeries([Point(0, 0)])
    s_almost_origin = GeoSeries([Point(1e-07, 0)])
    assert_geoseries_equal(s_origin, s_almost_origin, check_less_precise=True)
    with pytest.raises(AssertionError):
        assert_geoseries_equal(s_origin, s_almost_origin)