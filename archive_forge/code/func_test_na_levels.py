from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_na_levels():
    result = MultiIndex(levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[0, -1, 1, 2, 3, 4]])
    expected = MultiIndex(levels=[[np.nan, None, pd.NaT, 128, 2]], codes=[[-1, -1, -1, -1, 3, 4]])
    tm.assert_index_equal(result, expected)
    result = MultiIndex(levels=[[np.nan, 's', pd.NaT, 128, None]], codes=[[0, -1, 1, 2, 3, 4]])
    expected = MultiIndex(levels=[[np.nan, 's', pd.NaT, 128, None]], codes=[[-1, -1, 1, -1, 3, -1]])
    tm.assert_index_equal(result, expected)
    result = MultiIndex(levels=[[1, 2, 3, 4, 5]], codes=[[0, -1, 1, 2, 3, 4]]).set_levels([[np.nan, 's', pd.NaT, 128, None]])
    tm.assert_index_equal(result, expected)
    result = MultiIndex(levels=[[np.nan, 's', pd.NaT, 128, None]], codes=[[1, 2, 2, 2, 2, 2]]).set_codes([[0, -1, 1, 2, 3, 4]])
    tm.assert_index_equal(result, expected)