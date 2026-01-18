from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_inconsistent_return_type():
    df = DataFrame({'A': ['Tiger', 'Tiger', 'Tiger', 'Lamb', 'Lamb', 'Pony', 'Pony'], 'B': Series(np.arange(7), dtype='int64'), 'C': date_range('20130101', periods=7)})

    def f_0(grp):
        return grp.iloc[0]
    expected = df.groupby('A').first()[['B']]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(f_0)[['B']]
    tm.assert_frame_equal(result, expected)

    def f_1(grp):
        if grp.name == 'Tiger':
            return None
        return grp.iloc[0]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(f_1)[['B']]
    e = expected.copy()
    e.loc['Tiger'] = np.nan
    tm.assert_frame_equal(result, e)

    def f_2(grp):
        if grp.name == 'Pony':
            return None
        return grp.iloc[0]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(f_2)[['B']]
    e = expected.copy()
    e.loc['Pony'] = np.nan
    tm.assert_frame_equal(result, e)

    def f_3(grp):
        if grp.name == 'Pony':
            return None
        return grp.iloc[0]
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(f_3)[['C']]
    e = df.groupby('A').first()[['C']]
    e.loc['Pony'] = pd.NaT
    tm.assert_frame_equal(result, e)

    def f_4(grp):
        if grp.name == 'Pony':
            return None
        return grp.iloc[0].loc['C']
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(f_4)
    e = df.groupby('A').first()['C'].copy()
    e.loc['Pony'] = np.nan
    e.name = None
    tm.assert_series_equal(result, e)