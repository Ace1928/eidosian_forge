import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_getitem_setitem_multislice(self):
    levels = [['t1', 't2'], ['a', 'b', 'c']]
    codes = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
    midx = MultiIndex(codes=codes, levels=levels, names=[None, 'id'])
    df = DataFrame({'value': [1, 2, 3, 7, 8]}, index=midx)
    result = df.loc[:, 'value']
    tm.assert_series_equal(df['value'], result)
    result = df.loc[df.index[1:3], 'value']
    tm.assert_series_equal(df['value'][1:3], result)
    result = df.loc[:, :]
    tm.assert_frame_equal(df, result)
    result = df
    df.loc[:, 'value'] = 10
    result['value'] = 10
    tm.assert_frame_equal(df, result)
    df.loc[:, :] = 10
    tm.assert_frame_equal(df, result)