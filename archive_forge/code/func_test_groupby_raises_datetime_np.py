import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['agg', 'transform'])
@pytest.mark.parametrize('groupby_func_np', [np.sum, np.mean])
def test_groupby_raises_datetime_np(how, by, groupby_series, groupby_func_np, df_with_datetime_col):
    df = df_with_datetime_col
    gb = df.groupby(by=by)
    if groupby_series:
        gb = gb['d']
    klass, msg = {np.sum: (TypeError, 'datetime64 type does not support sum operations'), np.mean: (None, '')}[groupby_func_np]
    if groupby_series:
        warn_msg = 'using SeriesGroupBy.[sum|mean]'
    else:
        warn_msg = 'using DataFrameGroupBy.[sum|mean]'
    _call_and_check(klass, msg, how, gb, groupby_func_np, (), warn_msg=warn_msg)