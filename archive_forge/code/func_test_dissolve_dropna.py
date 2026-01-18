import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_dropna():
    gdf = geopandas.GeoDataFrame({'a': [1, 1, None], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)'])})
    expected_with_na = geopandas.GeoDataFrame({'a': [1.0, np.nan], 'geometry': geopandas.array.from_wkt(['MULTIPOINT (0 0, 1 1)', 'POINT (2 2)'])}).set_index('a')
    expected_no_na = geopandas.GeoDataFrame({'a': [1.0], 'geometry': geopandas.array.from_wkt(['MULTIPOINT (0 0, 1 1)'])}).set_index('a')
    assert_frame_equal(expected_with_na, gdf.dissolve('a', dropna=False))
    assert_frame_equal(expected_no_na, gdf.dissolve('a'))