import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_apply_convert_dtypes_keyword(s):
    if not compat.PANDAS_GE_21:
        recorder = warnings.catch_warnings(record=True)
    else:
        recorder = pytest.warns()
    with recorder as record:
        res = s.apply(lambda x: x, convert_dtype=True, args=())
    assert_geoseries_equal(res, s)
    if compat.PANDAS_GE_21:
        assert len(record) == 1
        assert 'the convert_dtype parameter' in str(record[0].message)
    else:
        assert len(record) == 0