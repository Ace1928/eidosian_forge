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
def test_array_accepts_pyarrow_array():
    arr = pa.array([1, 2, 3])
    result = pa.array(arr)
    assert arr == result
    result = pa.array(arr, type=pa.uint8())
    expected = pa.array([1, 2, 3], type=pa.uint8())
    assert expected == result
    assert expected.type == pa.uint8()
    arr = pa.array([2 ** 63 - 1], type=pa.int64())
    with pytest.raises(pa.ArrowInvalid):
        pa.array(arr, type=pa.int32())
    expected = pa.array([-1], type=pa.int32())
    result = pa.array(arr, type=pa.int32(), safe=False)
    assert result == expected
    result = pa.array(arr, memory_pool=pa.default_memory_pool())
    assert arr == result