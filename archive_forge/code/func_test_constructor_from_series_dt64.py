from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('klass', [Index, DatetimeIndex])
def test_constructor_from_series_dt64(self, klass):
    stamps = [Timestamp('20110101'), Timestamp('20120101'), Timestamp('20130101')]
    expected = DatetimeIndex(stamps)
    ser = Series(stamps)
    result = klass(ser)
    tm.assert_index_equal(result, expected)