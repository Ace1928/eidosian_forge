import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', [None, 'int64', 'Int64'])
@pytest.mark.parametrize('index', [None, [0, 1, 2]])
@pytest.mark.parametrize('columns', [None, ['a', 'b'], ['a', 'b', 'c']])
def test_dataframe_from_dict_of_series(request, using_copy_on_write, warn_copy_on_write, columns, index, dtype):
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    s1_orig = s1.copy()
    expected = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=index, columns=columns, dtype=dtype)
    result = DataFrame({'a': s1, 'b': s2}, index=index, columns=columns, dtype=dtype, copy=False)
    assert np.shares_memory(get_array(result, 'a'), get_array(s1))
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[0, 0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(s1))
        tm.assert_series_equal(s1, s1_orig)
    else:
        assert s1.iloc[0] == 10
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = DataFrame({'a': s1, 'b': s2}, index=index, columns=columns, dtype=dtype, copy=False)
    with tm.assert_cow_warning(warn_copy_on_write):
        s1.iloc[0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(s1))
        tm.assert_frame_equal(result, expected)
    else:
        assert result.iloc[0, 0] == 10