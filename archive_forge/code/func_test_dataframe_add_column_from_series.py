import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_add_column_from_series(backend, using_copy_on_write):
    _, DataFrame, Series = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    s = Series([10, 11, 12])
    df['new'] = s
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'new'), get_array(s))
    else:
        assert not np.shares_memory(get_array(df, 'new'), get_array(s))
    s[0] = 0
    expected = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'new': [10, 11, 12]})
    tm.assert_frame_equal(df, expected)