from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert_mismatched_tz(self):
    idx = date_range('1/1/2000', periods=3, freq='D', tz='Asia/Tokyo', name='idx')
    item = Timestamp('2000-01-04', tz='US/Eastern')
    result = idx.insert(3, item)
    expected = Index(list(idx[:3]) + [item.tz_convert(idx.tz)] + list(idx[3:]), name='idx')
    assert expected.dtype == idx.dtype
    tm.assert_index_equal(result, expected)
    item = datetime(2000, 1, 4, tzinfo=pytz.timezone('US/Eastern'))
    result = idx.insert(3, item)
    expected = Index(list(idx[:3]) + [item.astimezone(idx.tzinfo)] + list(idx[3:]), name='idx')
    assert expected.dtype == idx.dtype
    tm.assert_index_equal(result, expected)