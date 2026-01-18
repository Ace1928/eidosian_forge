from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_datetime_result_dtypes(using_infer_string):
    data = DataFrame.from_records([(pd.Timestamp(2016, 1, 1), 'red', 'dark', 1, '8'), (pd.Timestamp(2015, 1, 1), 'green', 'stormy', 2, '9'), (pd.Timestamp(2014, 1, 1), 'blue', 'bright', 3, '10'), (pd.Timestamp(2013, 1, 1), 'blue', 'calm', 4, 'potato')], columns=['observation', 'color', 'mood', 'intensity', 'score'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = data.groupby('color').apply(lambda g: g.iloc[0]).dtypes
    dtype = 'string' if using_infer_string else object
    expected = Series([np.dtype('datetime64[ns]'), dtype, dtype, np.int64, dtype], index=['observation', 'color', 'mood', 'intensity', 'score'])
    tm.assert_series_equal(result, expected)