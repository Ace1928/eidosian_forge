from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert3(self, unit):
    idx = date_range('1/1/2000', periods=3, freq='ME', name='idx', unit=unit)
    result = idx.insert(3, datetime(2000, 1, 2))
    expected = DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31', '2000-01-02'], name='idx', freq=None).as_unit(unit)
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert result.freq is None