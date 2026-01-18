from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_datetimelike_with_inference(self):
    one_hour = timedelta(hours=1)
    df = DataFrame(index=date_range('20130101', periods=4))
    df['A'] = np.array([1 * one_hour] * 4, dtype='m8[ns]')
    df.loc[:, 'B'] = np.array([2 * one_hour] * 4, dtype='m8[ns]')
    df.loc[df.index[:3], 'C'] = np.array([3 * one_hour] * 3, dtype='m8[ns]')
    df.loc[:, 'D'] = np.array([4 * one_hour] * 4, dtype='m8[ns]')
    df.loc[df.index[:3], 'E'] = np.array([5 * one_hour] * 3, dtype='m8[ns]')
    df['F'] = np.timedelta64('NaT')
    df.loc[df.index[:-1], 'F'] = np.array([6 * one_hour] * 3, dtype='m8[ns]')
    df.loc[df.index[-3]:, 'G'] = date_range('20130101', periods=3)
    df['H'] = np.datetime64('NaT')
    result = df.dtypes
    expected = Series([np.dtype('timedelta64[ns]')] * 6 + [np.dtype('datetime64[ns]')] * 2, index=list('ABCDEFGH'))
    tm.assert_series_equal(result, expected)