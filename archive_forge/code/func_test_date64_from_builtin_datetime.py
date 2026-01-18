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
def test_date64_from_builtin_datetime():
    val1 = datetime.datetime(2000, 1, 1, 12, 34, 56, 123456)
    val2 = datetime.datetime(2000, 1, 1)
    result = pa.array([val1, val2], type='date64')
    result2 = pa.array([val1.date(), val2.date()], type='date64')
    assert result.equals(result2)
    as_i8 = result.view('int64')
    assert as_i8[0].as_py() == as_i8[1].as_py()