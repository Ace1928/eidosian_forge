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
def test_interval_array_from_timedelta():
    data = [None, datetime.timedelta(days=1, seconds=1, microseconds=1, milliseconds=1, minutes=1, hours=1, weeks=1)]
    arr = pa.array(data, pa.month_day_nano_interval())
    assert isinstance(arr, pa.MonthDayNanoIntervalArray)
    assert arr.type == pa.month_day_nano_interval()
    expected_list = [None, pa.MonthDayNano([0, 8, datetime.timedelta(seconds=1, microseconds=1, milliseconds=1, minutes=1, hours=1) // datetime.timedelta(microseconds=1) * 1000])]
    expected = pa.array(expected_list)
    assert arr.equals(expected)
    assert arr.to_pylist() == expected_list