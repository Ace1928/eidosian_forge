import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_datetime_subclassing():
    data = [MyDate(2007, 7, 13)]
    date_type = pa.date32()
    arr_date = pa.array(data, type=date_type)
    assert len(arr_date) == 1
    assert arr_date.type == date_type
    assert arr_date[0].as_py() == datetime.date(2007, 7, 13)
    data = [MyDatetime(2007, 7, 13, 1, 23, 34, 123456)]
    s = pa.timestamp('s')
    ms = pa.timestamp('ms')
    us = pa.timestamp('us')
    arr_s = pa.array(data, type=s)
    assert len(arr_s) == 1
    assert arr_s.type == s
    assert arr_s[0].as_py() == datetime.datetime(2007, 7, 13, 1, 23, 34, 0)
    arr_ms = pa.array(data, type=ms)
    assert len(arr_ms) == 1
    assert arr_ms.type == ms
    assert arr_ms[0].as_py() == datetime.datetime(2007, 7, 13, 1, 23, 34, 123000)
    arr_us = pa.array(data, type=us)
    assert len(arr_us) == 1
    assert arr_us.type == us
    assert arr_us[0].as_py() == datetime.datetime(2007, 7, 13, 1, 23, 34, 123456)
    data = [MyTimedelta(123, 456, 1002)]
    s = pa.duration('s')
    ms = pa.duration('ms')
    us = pa.duration('us')
    arr_s = pa.array(data)
    assert len(arr_s) == 1
    assert arr_s.type == us
    assert arr_s[0].as_py() == datetime.timedelta(123, 456, 1002)
    arr_s = pa.array(data, type=s)
    assert len(arr_s) == 1
    assert arr_s.type == s
    assert arr_s[0].as_py() == datetime.timedelta(123, 456)
    arr_ms = pa.array(data, type=ms)
    assert len(arr_ms) == 1
    assert arr_ms.type == ms
    assert arr_ms[0].as_py() == datetime.timedelta(123, 456, 1000)
    arr_us = pa.array(data, type=us)
    assert len(arr_us) == 1
    assert arr_us.type == us
    assert arr_us[0].as_py() == datetime.timedelta(123, 456, 1002)