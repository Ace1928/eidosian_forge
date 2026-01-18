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
def test_array_from_scalar():
    pytz = pytest.importorskip('pytz')
    today = datetime.date.today()
    now = datetime.datetime.now()
    now_utc = now.replace(tzinfo=pytz.utc)
    now_with_tz = now_utc.astimezone(pytz.timezone('US/Eastern'))
    oneday = datetime.timedelta(days=1)
    cases = [(None, 1, pa.array([None])), (None, 10, pa.nulls(10)), (-1, 3, pa.array([-1, -1, -1], type=pa.int64())), (2.71, 2, pa.array([2.71, 2.71], type=pa.float64())), ('string', 4, pa.array(['string'] * 4)), (pa.scalar(8, type=pa.uint8()), 17, pa.array([8] * 17, type=pa.uint8())), (pa.scalar(None), 3, pa.array([None, None, None])), (pa.scalar(True), 11, pa.array([True] * 11)), (today, 2, pa.array([today] * 2)), (now, 10, pa.array([now] * 10)), (now_with_tz, 2, pa.array([now_utc] * 2, type=pa.timestamp('us', tz=pytz.timezone('US/Eastern')))), (now.time(), 9, pa.array([now.time()] * 9)), (oneday, 4, pa.array([oneday] * 4)), (False, 9, pa.array([False] * 9)), ([1, 2], 2, pa.array([[1, 2], [1, 2]])), (pa.scalar([-1, 3], type=pa.large_list(pa.int8())), 5, pa.array([[-1, 3]] * 5, type=pa.large_list(pa.int8()))), ({'a': 1, 'b': 2}, 3, pa.array([{'a': 1, 'b': 2}] * 3))]
    for value, size, expected in cases:
        arr = pa.repeat(value, size)
        assert len(arr) == size
        assert arr.type.equals(expected.type)
        assert arr.equals(expected)
        if expected.type == pa.null():
            assert arr.null_count == size
        else:
            assert arr.null_count == 0