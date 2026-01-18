import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_infer_objects_no_reference(using_copy_on_write):
    df = DataFrame({'a': [1, 2], 'b': 'c', 'c': 1, 'd': Series([Timestamp('2019-12-31'), Timestamp('2020-12-31')], dtype='object'), 'e': 'b'})
    df = df.infer_objects()
    arr_a = get_array(df, 'a')
    arr_b = get_array(df, 'b')
    arr_d = get_array(df, 'd')
    df.iloc[0, 0] = 0
    df.iloc[0, 1] = 'd'
    df.iloc[0, 3] = Timestamp('2018-12-31')
    if using_copy_on_write:
        assert np.shares_memory(arr_a, get_array(df, 'a'))
        assert not np.shares_memory(arr_b, get_array(df, 'b'))
        assert np.shares_memory(arr_d, get_array(df, 'd'))