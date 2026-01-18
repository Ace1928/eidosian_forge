from __future__ import absolute_import
from itertools import product
import json
from packaging.version import Version
import os
import pathlib
import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np
import pyproj
import shapely
from shapely.geometry import box, Point, MultiPolygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
@pytest.mark.skipif(Version(pyarrow.__version__) < Version('0.17.0'), reason='Feather only supported for pyarrow >= 0.17')
@pytest.mark.parametrize('compression', ['uncompressed', 'lz4', 'zstd'])
def test_feather_compression(compression, tmpdir):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """
    test_dataset = 'naturalearth_lowres'
    df = read_file(get_path(test_dataset))
    filename = os.path.join(str(tmpdir), 'test.feather')
    df.to_feather(filename, compression=compression)
    pq_df = read_feather(filename)
    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)