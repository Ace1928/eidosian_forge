import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', [lambda df: df.values, lambda df: np.asarray(df)], ids=['values', 'asarray'])
def test_dataframe_values(using_copy_on_write, using_array_manager, method):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df_orig = df.copy()
    arr = method(df)
    if using_copy_on_write:
        assert np.shares_memory(arr, get_array(df, 'a'))
        assert arr.flags.writeable is False
        with pytest.raises(ValueError, match='read-only'):
            arr[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)
        df.iloc[0, 0] = 0
        assert df.values[0, 0] == 0
    else:
        assert arr.flags.writeable is True
        arr[0, 0] = 0
        if not using_array_manager:
            assert df.iloc[0, 0] == 0
        else:
            tm.assert_frame_equal(df, df_orig)