import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    df = frame.T.copy()
    values = df.values.copy()
    result = df[df > 0]
    expected = df.where(df > 0)
    tm.assert_frame_equal(result, expected)
    df[df > 0] = 5
    values[values > 0] = 5
    tm.assert_almost_equal(df.values, values)
    df[df == 5] = 0
    values[values == 5] = 0
    tm.assert_almost_equal(df.values, values)
    df[df[:-1] < 0] = 2
    np.putmask(values[:-1], values[:-1] < 0, 2)
    tm.assert_almost_equal(df.values, values)
    with pytest.raises(TypeError, match='boolean values only'):
        df[df * 0] = 2