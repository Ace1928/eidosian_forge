import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_dict_of_series(self):
    data = {'A': pd.Series(range(3)), 'B': pd.Series(np.arange(3.0)), 'geometry': GeoSeries([Point(x, x) for x in range(3)])}
    df = GeoDataFrame(data)
    check_geodataframe(df)
    df = GeoDataFrame(data, index=pd.Index([1, 2]))
    check_geodataframe(df)
    assert_index_equal(df.index, pd.Index([1, 2]))
    assert df['A'].tolist() == [1, 2]
    data = {'A': pd.Series(range(3)), 'B': np.arange(3.0), 'geometry': GeoSeries([Point(x, x) for x in range(3)])}
    with pytest.raises(ValueError):
        GeoDataFrame(data, index=[1, 2])