import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
@pytest.mark.skipif(not PANDAS_GE_15 or PANDAS_GE_20, reason='warning for pandas 1.5.x')
def test_mean_dissolve_warning_capture(nybb_polydf, first, expected_mean):
    with pytest.warns(FutureWarning, match='.*used in dissolve is deprecated.*'):
        nybb_polydf.dissolve('manhattan_bronx', aggfunc='mean')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nybb_polydf.dissolve('manhattan_bronx', aggfunc='first')