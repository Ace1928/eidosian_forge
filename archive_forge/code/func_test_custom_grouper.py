from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_custom_grouper(index, unit):
    dti = index.as_unit(unit)
    s = Series(np.array([1] * len(dti)), index=dti, dtype='int64')
    b = Grouper(freq=Minute(5))
    g = s.groupby(b)
    g.ohlc()
    funcs = ['sum', 'mean', 'prod', 'min', 'max', 'var']
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)
    b = Grouper(freq=Minute(5), closed='right', label='right')
    g = s.groupby(b)
    g.ohlc()
    funcs = ['sum', 'mean', 'prod', 'min', 'max', 'var']
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)
    assert g.ngroups == 2593
    assert notna(g.mean()).all()
    arr = [1] + [5] * 2592
    idx = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq='5min').as_unit(unit)
    expect = Series(arr, index=idx)
    result = g.agg('sum')
    tm.assert_series_equal(result, expect)