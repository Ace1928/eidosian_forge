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
@pytest.mark.parametrize('t,check_func', [(pa.date32(), types.is_date32), (pa.date64(), types.is_date64), (pa.time32('s'), types.is_time32), (pa.time64('ns'), types.is_time64), (pa.int8(), types.is_int8), (pa.int16(), types.is_int16), (pa.int32(), types.is_int32), (pa.int64(), types.is_int64), (pa.uint8(), types.is_uint8), (pa.uint16(), types.is_uint16), (pa.uint32(), types.is_uint32), (pa.uint64(), types.is_uint64), (pa.float16(), types.is_float16), (pa.float32(), types.is_float32), (pa.float64(), types.is_float64)])
def test_exact_primitive_types(t, check_func):
    assert check_func(t)