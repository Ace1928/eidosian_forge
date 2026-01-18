from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')], ids=['datetime64', 'datetime64tz'])
@pytest.mark.parametrize('insert_value', [pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01', tz='Asia/Tokyo'), 1])
def test_insert_index_datetimes(self, fill_val, exp_dtype, insert_value):
    obj = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
    assert obj.dtype == exp_dtype
    exp = pd.DatetimeIndex(['2011-01-01', fill_val.date(), '2011-01-02', '2011-01-03', '2011-01-04'], tz=fill_val.tz).as_unit('ns')
    self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)
    if fill_val.tz:
        ts = pd.Timestamp('2012-01-01')
        result = obj.insert(1, ts)
        expected = obj.astype(object).insert(1, ts)
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)
        ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
        result = obj.insert(1, ts)
        expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
        assert expected.dtype == obj.dtype
        tm.assert_index_equal(result, expected)
    else:
        ts = pd.Timestamp('2012-01-01', tz='Asia/Tokyo')
        result = obj.insert(1, ts)
        expected = obj.astype(object).insert(1, ts)
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)
    item = 1
    result = obj.insert(1, item)
    expected = obj.astype(object).insert(1, item)
    assert expected[1] == item
    assert expected.dtype == object
    tm.assert_index_equal(result, expected)