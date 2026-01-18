from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('kind', ['datetime', 'period'])
def test_groupby_resample_kind(kind):
    df = DataFrame({'datetime': pd.to_datetime(['20181101 1100', '20181101 1200', '20181102 1300', '20181102 1400']), 'group': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
    df = df.set_index('datetime')
    result = df.groupby('group')['value'].resample('D', kind=kind).last()
    dt_level = pd.DatetimeIndex(['2018-11-01', '2018-11-02'])
    if kind == 'period':
        dt_level = dt_level.to_period(freq='D')
    expected_index = pd.MultiIndex.from_product([['A', 'B'], dt_level], names=['group', 'datetime'])
    expected = Series([1, 3, 2, 4], index=expected_index, name='value')
    tm.assert_series_equal(result, expected)