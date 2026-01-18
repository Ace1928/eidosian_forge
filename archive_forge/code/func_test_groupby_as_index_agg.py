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
def test_groupby_as_index_agg(df):
    grouped = df.groupby('A', as_index=False)
    result = grouped[['C', 'D']].agg('mean')
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)
    result2 = grouped.agg({'C': 'mean', 'D': 'sum'})
    expected2 = grouped.mean(numeric_only=True)
    expected2['D'] = grouped.sum()['D']
    tm.assert_frame_equal(result2, expected2)
    grouped = df.groupby('A', as_index=True)
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped['C'].agg({'Q': 'sum'})
    grouped = df.groupby(['A', 'B'], as_index=False)
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)
    result2 = grouped.agg({'C': 'mean', 'D': 'sum'})
    expected2 = grouped.mean()
    expected2['D'] = grouped.sum()['D']
    tm.assert_frame_equal(result2, expected2)
    expected3 = grouped['C'].sum()
    expected3 = DataFrame(expected3).rename(columns={'C': 'Q'})
    msg = 'Passing a dictionary to SeriesGroupBy.agg is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result3 = grouped['C'].agg({'Q': 'sum'})
    tm.assert_frame_equal(result3, expected3)
    df = DataFrame(np.random.default_rng(2).integers(0, 100, (50, 3)), columns=['jim', 'joe', 'jolie'])
    ts = Series(np.random.default_rng(2).integers(5, 10, 50), name='jim')
    gr = df.groupby(ts)
    gr.nth(0)
    msg = 'The behavior of DataFrame.sum with axis=None is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        res = gr.apply(sum)
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        alt = df.groupby(ts).apply(sum)
    tm.assert_frame_equal(res, alt)
    for attr in ['mean', 'max', 'count', 'idxmax', 'cumsum', 'all']:
        gr = df.groupby(ts, as_index=False)
        left = getattr(gr, attr)()
        gr = df.groupby(ts.values, as_index=True)
        right = getattr(gr, attr)().reset_index(drop=True)
        tm.assert_frame_equal(left, right)