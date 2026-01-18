import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_multicolumn_dissolve(nybb_polydf, first):
    multi = nybb_polydf.copy()
    multi['dup_col'] = multi.manhattan_bronx
    multi_test = multi.dissolve(['manhattan_bronx', 'dup_col'], aggfunc='first')
    first_copy = first.copy()
    first_copy['dup_col'] = first_copy.index
    first_copy = first_copy.set_index([first_copy.index, 'dup_col'])
    assert_frame_equal(multi_test, first_copy, check_column_type=False)