import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_geoseries():
    assert_geoseries_equal(s1, s2)
    assert_geoseries_equal(s1, s3, check_series_type=False, check_dtype=False)
    assert_geoseries_equal(s3, s2, check_series_type=False, check_dtype=False)
    assert_geoseries_equal(s1, s4, check_series_type=False)
    with pytest.raises(AssertionError) as error:
        assert_geoseries_equal(s1, s2, check_less_precise=True)
    assert '1 out of 2 geometries are not almost equal' in str(error.value)
    assert 'not almost equal: [0]' in str(error.value)
    with pytest.raises(AssertionError) as error:
        assert_geoseries_equal(s2, s6, check_less_precise=False)
    assert '1 out of 2 geometries are not equal' in str(error.value)
    assert 'not equal: [0]' in str(error.value)