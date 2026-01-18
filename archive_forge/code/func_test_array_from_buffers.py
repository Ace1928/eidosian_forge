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
def test_array_from_buffers():
    values_buf = pa.py_buffer(np.int16([4, 5, 6, 7]))
    nulls_buf = pa.py_buffer(np.uint8([13]))
    arr = pa.Array.from_buffers(pa.int16(), 4, [nulls_buf, values_buf])
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [4, None, 6, 7]
    arr = pa.Array.from_buffers(pa.int16(), 4, [None, values_buf])
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [4, 5, 6, 7]
    arr = pa.Array.from_buffers(pa.int16(), 3, [nulls_buf, values_buf], offset=1)
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [None, 6, 7]
    with pytest.raises(TypeError):
        pa.Array.from_buffers(pa.int16(), 3, ['', ''], offset=1)