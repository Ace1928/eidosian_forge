import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_level():
    gdf = geopandas.GeoDataFrame({'a': [1, 1, 2, 2], 'b': [3, 4, 4, 4], 'c': [3, 4, 5, 6], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)'])}).set_index(['a', 'b', 'c'])
    expected_a = geopandas.GeoDataFrame({'a': [1, 2], 'geometry': geopandas.array.from_wkt(['MULTIPOINT (0 0, 1 1)', 'MULTIPOINT (2 2, 3 3)'])}).set_index('a')
    expected_b = geopandas.GeoDataFrame({'b': [3, 4], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'MULTIPOINT (1 1, 2 2, 3 3)'])}).set_index('b')
    expected_ab = geopandas.GeoDataFrame({'a': [1, 1, 2], 'b': [3, 4, 4], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'MULTIPOINT (2 2, 3 3)'])}).set_index(['a', 'b'])
    assert_frame_equal(expected_a, gdf.dissolve(level=0))
    assert_frame_equal(expected_a, gdf.dissolve(level='a'))
    assert_frame_equal(expected_b, gdf.dissolve(level=1))
    assert_frame_equal(expected_b, gdf.dissolve(level='b'))
    assert_frame_equal(expected_ab, gdf.dissolve(level=[0, 1]))
    assert_frame_equal(expected_ab, gdf.dissolve(level=['a', 'b']))