import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['method', 'agg', 'transform'])
def test_groupby_raises_datetime(how, by, groupby_series, groupby_func, df_with_datetime_col):
    df = df_with_datetime_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)
    if groupby_series:
        gb = gb['d']
        if groupby_func == 'corrwith':
            assert not hasattr(gb, 'corrwith')
            return
    klass, msg = {'all': (None, ''), 'any': (None, ''), 'bfill': (None, ''), 'corrwith': (TypeError, 'cannot perform __mul__ with this index type'), 'count': (None, ''), 'cumcount': (None, ''), 'cummax': (None, ''), 'cummin': (None, ''), 'cumprod': (TypeError, 'datetime64 type does not support cumprod operations'), 'cumsum': (TypeError, 'datetime64 type does not support cumsum operations'), 'diff': (None, ''), 'ffill': (None, ''), 'fillna': (None, ''), 'first': (None, ''), 'idxmax': (None, ''), 'idxmin': (None, ''), 'last': (None, ''), 'max': (None, ''), 'mean': (None, ''), 'median': (None, ''), 'min': (None, ''), 'ngroup': (None, ''), 'nunique': (None, ''), 'pct_change': (TypeError, 'cannot perform __truediv__ with this index type'), 'prod': (TypeError, 'datetime64 type does not support prod'), 'quantile': (None, ''), 'rank': (None, ''), 'sem': (None, ''), 'shift': (None, ''), 'size': (None, ''), 'skew': (TypeError, '|'.join(['dtype datetime64\\[ns\\] does not support reduction', 'datetime64 type does not support skew operations'])), 'std': (None, ''), 'sum': (TypeError, 'datetime64 type does not support sum operations'), 'var': (TypeError, 'datetime64 type does not support var operations')}[groupby_func]
    if groupby_func in ['any', 'all']:
        warn_msg = f"'{groupby_func}' with datetime64 dtypes is deprecated"
    elif groupby_func == 'fillna':
        kind = 'Series' if groupby_series else 'DataFrame'
        warn_msg = f'{kind}GroupBy.fillna is deprecated'
    else:
        warn_msg = ''
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg=warn_msg)