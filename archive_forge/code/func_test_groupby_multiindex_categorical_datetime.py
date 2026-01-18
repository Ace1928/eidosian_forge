from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_multiindex_categorical_datetime():
    df = DataFrame({'key1': Categorical(list('abcbabcba')), 'key2': Categorical(list(pd.date_range('2018-06-01 00', freq='1min', periods=3)) * 3), 'values': np.arange(9)})
    result = df.groupby(['key1', 'key2'], observed=False).mean()
    idx = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(pd.date_range('2018-06-01 00', freq='1min', periods=3))], names=['key1', 'key2'])
    expected = DataFrame({'values': [0, 4, 8, 3, 4, 5, 6, np.nan, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)