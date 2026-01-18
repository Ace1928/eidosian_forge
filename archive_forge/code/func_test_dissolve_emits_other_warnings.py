import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_emits_other_warnings(nybb_polydf):

    def sum_and_warn(group):
        warnings.warn('foo')
        if PANDAS_GE_20:
            return group.sum(numeric_only=False)
        else:
            return group.sum()
    with pytest.warns(UserWarning, match='foo'):
        nybb_polydf.dissolve('manhattan_bronx', aggfunc=sum_and_warn)