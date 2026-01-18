from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('dropna, expected', [(True, {'A': [12], 'B': [10.0], 'C': [1.0], 'D': ['a'], 'E': Categorical(['a'], categories=['a']), 'F': DatetimeIndex(['2000-01-02'], dtype='M8[ns]'), 'G': to_timedelta(['1 days'])}), (False, {'A': [12], 'B': [10.0], 'C': [np.nan], 'D': np.array([np.nan], dtype=object), 'E': Categorical([np.nan], categories=['a']), 'F': DatetimeIndex([pd.NaT], dtype='M8[ns]'), 'G': to_timedelta([pd.NaT])}), (True, {'H': [8, 9, np.nan, np.nan], 'I': [8, 9, np.nan, np.nan], 'J': [1, np.nan, np.nan, np.nan], 'K': Categorical(['a', np.nan, np.nan, np.nan], categories=['a']), 'L': DatetimeIndex(['2000-01-02', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['1 days', 'nan', 'nan', 'nan']), 'N': [0, 1, 2, 3]}), (False, {'H': [8, 9, np.nan, np.nan], 'I': [8, 9, np.nan, np.nan], 'J': [1, np.nan, np.nan, np.nan], 'K': Categorical([np.nan, 'a', np.nan, np.nan], categories=['a']), 'L': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['nan', '1 days', 'nan', 'nan']), 'N': [0, 1, 2, 3]})])
def test_mode_dropna(self, dropna, expected):
    df = DataFrame({'A': [12, 12, 19, 11], 'B': [10, 10, np.nan, 3], 'C': [1, np.nan, np.nan, np.nan], 'D': Series([np.nan, np.nan, 'a', np.nan], dtype=object), 'E': Categorical([np.nan, np.nan, 'a', np.nan]), 'F': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'G': to_timedelta(['1 days', 'nan', 'nan', 'nan']), 'H': [8, 8, 9, 9], 'I': [9, 9, 8, 8], 'J': [1, 1, np.nan, np.nan], 'K': Categorical(['a', np.nan, 'a', np.nan]), 'L': DatetimeIndex(['2000-01-02', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['1 days', 'nan', '1 days', 'nan']), 'N': np.arange(4, dtype='int64')})
    result = df[sorted(expected.keys())].mode(dropna=dropna)
    expected = DataFrame(expected)
    tm.assert_frame_equal(result, expected)