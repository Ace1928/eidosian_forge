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
def test_groupby_groups(df):
    g = df.groupby('value2')
    res = g.get_group(1)
    assert isinstance(res, GeoDataFrame)
    exp = df.loc[[0, 2]]
    assert_frame_equal(res, exp)