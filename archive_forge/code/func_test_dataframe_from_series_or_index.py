import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('cons', [Series, Index])
@pytest.mark.parametrize('data, dtype', [([1, 2], None), ([1, 2], 'int64'), (['a', 'b'], None)])
def test_dataframe_from_series_or_index(using_copy_on_write, warn_copy_on_write, data, dtype, cons):
    obj = cons(data, dtype=dtype)
    obj_orig = obj.copy()
    df = DataFrame(obj, dtype=dtype)
    assert np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = data[-1]
    if using_copy_on_write:
        tm.assert_equal(obj, obj_orig)