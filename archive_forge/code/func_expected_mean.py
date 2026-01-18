import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
@pytest.fixture
def expected_mean(merged_shapes):
    test_mean = merged_shapes.copy()
    test_mean['BoroCode'] = [4, 1.5]
    return test_mean