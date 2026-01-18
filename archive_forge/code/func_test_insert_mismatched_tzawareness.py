from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert_mismatched_tzawareness(self):
    idx = date_range('1/1/2000', periods=3, freq='D', tz='Asia/Tokyo', name='idx')
    item = Timestamp('2000-01-04')
    result = idx.insert(3, item)
    expected = Index(list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name='idx')
    tm.assert_index_equal(result, expected)
    item = datetime(2000, 1, 4)
    result = idx.insert(3, item)
    expected = Index(list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name='idx')
    tm.assert_index_equal(result, expected)