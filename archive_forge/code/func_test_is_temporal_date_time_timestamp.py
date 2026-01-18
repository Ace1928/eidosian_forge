from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_is_temporal_date_time_timestamp():
    date_types = [pa.date32(), pa.date64()]
    time_types = [pa.time32('s'), pa.time64('ns')]
    timestamp_types = [pa.timestamp('ms')]
    duration_types = [pa.duration('ms')]
    interval_types = [pa.month_day_nano_interval()]
    for case in date_types + time_types + timestamp_types + duration_types + interval_types:
        assert types.is_temporal(case)
    for case in date_types:
        assert types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)
    for case in time_types:
        assert types.is_time(case)
        assert not types.is_date(case)
        assert not types.is_timestamp(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)
    for case in timestamp_types:
        assert types.is_timestamp(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)
    for case in duration_types:
        assert types.is_duration(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)
        assert not types.is_interval(case)
    for case in interval_types:
        assert types.is_interval(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)
    assert not types.is_temporal(pa.int32())