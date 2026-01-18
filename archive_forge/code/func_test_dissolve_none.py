import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_none(nybb_polydf):
    test = nybb_polydf.dissolve(by=None)
    expected = GeoDataFrame({nybb_polydf.geometry.name: [nybb_polydf.geometry.unary_union], 'BoroName': ['Staten Island'], 'BoroCode': [5], 'manhattan_bronx': [5]}, geometry=nybb_polydf.geometry.name, crs=nybb_polydf.crs)
    assert_frame_equal(expected, test, check_column_type=False)