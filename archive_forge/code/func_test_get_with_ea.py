import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arr', [np.random.default_rng(2).standard_normal(10), DatetimeIndex(date_range('2020-01-01', periods=10), name='a').tz_localize(tz='US/Eastern')])
def test_get_with_ea(arr):
    ser = Series(arr, index=[2 * i for i in range(len(arr))])
    assert ser.get(4) == ser.iloc[2]
    result = ser.get([4, 6])
    expected = ser.iloc[[2, 3]]
    tm.assert_series_equal(result, expected)
    result = ser.get(slice(2))
    expected = ser.iloc[[0, 1]]
    tm.assert_series_equal(result, expected)
    assert ser.get(-1) is None
    assert ser.get(ser.index.max() + 1) is None
    ser = Series(arr[:6], index=list('abcdef'))
    assert ser.get('c') == ser.iloc[2]
    result = ser.get(slice('b', 'd'))
    expected = ser.iloc[[1, 2, 3]]
    tm.assert_series_equal(result, expected)
    result = ser.get('Z')
    assert result is None
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(4) == ser.iloc[4]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(-1) == ser.iloc[-1]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(len(ser)) is None
    ser = Series(arr)
    ser2 = ser[::2]
    assert ser2.get(1) is None