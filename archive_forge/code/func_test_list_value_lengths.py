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
@pytest.mark.parametrize(('offset_type', 'list_type'), [(pa.int32(), pa.list_(pa.int32())), (pa.int32(), pa.list_(pa.int32(), list_size=2)), (pa.int64(), pa.large_list(pa.int32()))])
def test_list_value_lengths(offset_type, list_type):
    if getattr(list_type, 'list_size', None):
        arr = pa.array([[0, 1], None, [None, None], [3, 4]], type=list_type)
        expected = pa.array([2, None, 2, 2], type=offset_type)
    else:
        arr = pa.array([[0, 1, 2], None, [], [3, 4]], type=list_type)
        expected = pa.array([3, None, 0, 2], type=offset_type)
    assert arr.value_lengths().equals(expected)