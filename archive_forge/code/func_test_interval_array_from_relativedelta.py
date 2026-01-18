from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_interval_array_from_relativedelta():
    from dateutil.relativedelta import relativedelta
    from pandas import DateOffset
    data = [None, relativedelta(years=1, months=1, days=1, seconds=1, microseconds=1, minutes=1, hours=1, weeks=1, leapdays=1)]
    arr = pa.array(data)
    assert isinstance(arr, pa.MonthDayNanoIntervalArray)
    assert arr.type == pa.month_day_nano_interval()
    expected_list = [None, pa.MonthDayNano([13, 8, datetime.timedelta(seconds=1, microseconds=1, minutes=1, hours=1) // datetime.timedelta(microseconds=1) * 1000])]
    expected = pa.array(expected_list)
    assert arr.equals(expected)
    assert arr.to_pandas().tolist() == [None, DateOffset(months=13, days=8, microseconds=datetime.timedelta(seconds=1, microseconds=1, minutes=1, hours=1) // datetime.timedelta(microseconds=1), nanoseconds=0)]
    with pytest.raises(ValueError):
        pa.array([DateOffset(years=(1 << 32) // 12, months=100)])
    with pytest.raises(ValueError):
        pa.array([DateOffset(weeks=(1 << 32) // 7, days=100)])
    with pytest.raises(ValueError):
        pa.array([DateOffset(seconds=(1 << 64) // 1000000000, nanoseconds=1)])
    with pytest.raises(ValueError):
        pa.array([DateOffset(microseconds=(1 << 64) // 100)])