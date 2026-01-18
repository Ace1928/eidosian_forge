import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', [None, 'int64'])
def test_dataframe_from_dict_of_series_with_reindex(dtype):
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    df = DataFrame({'a': s1, 'b': s2}, index=[1, 2, 3], dtype=dtype, copy=False)
    arr_before = get_array(df, 'a')
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, 'a')
    assert np.shares_memory(arr_before, arr_after)