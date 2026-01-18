from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_index_date_object(using_infer_string):
    ts = ['2011-05-16 00:00', '2011-05-16 01:00', '2011-05-16 02:00', '2011-05-16 03:00', '2011-05-17 02:00', '2011-05-17 03:00', '2011-05-17 04:00', '2011-05-17 05:00', '2011-05-18 02:00', '2011-05-18 03:00', '2011-05-18 04:00', '2011-05-18 05:00']
    df = DataFrame([row.split() for row in ts], columns=['date', 'time'])
    df['value'] = [1.40893, 1.4076, 1.4075, 1.40649, 1.40893, 1.4076, 1.4075, 1.40649, 1.40893, 1.4076, 1.4075, 1.40649]
    dtype = 'string[pyarrow_numpy]' if using_infer_string else object
    exp_idx = Index(['2011-05-16', '2011-05-17', '2011-05-18'], dtype=dtype, name='date')
    expected = Series(['00:00', '02:00', '02:00'], index=exp_idx)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('date', group_keys=False).apply(lambda x: x['time'][x['value'].idxmax()])
    tm.assert_series_equal(result, expected)