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
def test_groupby_nat_exclude():
    df = DataFrame({'values': np.random.default_rng(2).standard_normal(8), 'dt': [np.nan, Timestamp('2013-01-01'), np.nan, Timestamp('2013-02-01'), np.nan, Timestamp('2013-02-01'), np.nan, Timestamp('2013-01-01')], 'str': [np.nan, 'a', np.nan, 'a', np.nan, 'a', np.nan, 'b']})
    grouped = df.groupby('dt')
    expected = [Index([1, 7]), Index([3, 5])]
    keys = sorted(grouped.groups.keys())
    assert len(keys) == 2
    for k, e in zip(keys, expected):
        tm.assert_index_equal(grouped.groups[k], e)
    tm.assert_frame_equal(grouped._grouper.groupings[0].obj, df)
    assert grouped.ngroups == 2
    expected = {Timestamp('2013-01-01 00:00:00'): np.array([1, 7], dtype=np.intp), Timestamp('2013-02-01 00:00:00'): np.array([3, 5], dtype=np.intp)}
    for k in grouped.indices:
        tm.assert_numpy_array_equal(grouped.indices[k], expected[k])
    tm.assert_frame_equal(grouped.get_group(Timestamp('2013-01-01')), df.iloc[[1, 7]])
    tm.assert_frame_equal(grouped.get_group(Timestamp('2013-02-01')), df.iloc[[3, 5]])
    with pytest.raises(KeyError, match='^NaT$'):
        grouped.get_group(pd.NaT)
    nan_df = DataFrame({'nan': [np.nan, np.nan, np.nan], 'nat': [pd.NaT, pd.NaT, pd.NaT]})
    assert nan_df['nan'].dtype == 'float64'
    assert nan_df['nat'].dtype == 'datetime64[ns]'
    for key in ['nan', 'nat']:
        grouped = nan_df.groupby(key)
        assert grouped.groups == {}
        assert grouped.ngroups == 0
        assert grouped.indices == {}
        with pytest.raises(KeyError, match='^nan$'):
            grouped.get_group(np.nan)
        with pytest.raises(KeyError, match='^NaT$'):
            grouped.get_group(pd.NaT)