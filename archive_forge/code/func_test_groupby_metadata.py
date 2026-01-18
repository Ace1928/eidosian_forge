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
@pytest.mark.skip_no_sindex
@pytest.mark.parametrize('crs', [None, 'EPSG:4326'])
@pytest.mark.parametrize('geometry_name', ['geometry', 'geom'])
def test_groupby_metadata(crs, geometry_name):
    df = GeoDataFrame({geometry_name: [Point(0, 0), Point(1, 1), Point(0, 0)], 'value1': np.arange(3, dtype='int64'), 'value2': np.array([1, 2, 1], dtype='int64')}, crs=crs, geometry=geometry_name)
    kwargs = {}
    if compat.PANDAS_GE_22:
        kwargs = dict(include_groups=False)

    def func(group):
        assert isinstance(group, GeoDataFrame)
        assert group.crs == crs
    df.groupby('value2').apply(func, **kwargs)
    if (compat.PANDAS_GE_21 if geometry_name == 'geometry' else compat.PANDAS_GE_20) and (not compat.PANDAS_GE_22):
        with pytest.raises(AttributeError):
            df.groupby('value2')[[geometry_name, 'value1']].apply(func)
    else:
        df.groupby('value2')[[geometry_name, 'value1']].apply(func)
    res = df.groupby('value2').apply(lambda x: geopandas.sjoin(x, x[[geometry_name, 'value1']], how='inner'), **kwargs)
    if compat.PANDAS_GE_22:
        take_indices = [0, 0, 2, 2, 1]
        value_right = [0, 2, 0, 2, 1]
    else:
        take_indices = [0, 2, 0, 2, 1]
        value_right = [0, 0, 2, 2, 1]
    expected = df.take(take_indices).set_index('value2', drop=compat.PANDAS_GE_22, append=True).swaplevel().rename(columns={'value1': 'value1_left'}).assign(value1_right=value_right)
    assert_geodataframe_equal(res.drop(columns=['index_right']), expected)