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
@pytest.mark.parametrize('list_type_factory', [pa.list_, pa.large_list])
def test_list_array_flatten_non_canonical(list_type_factory):
    typ = list_type_factory(pa.int64())
    arr = pa.array([[1], [2, 3], [4, 5, 6]], type=typ)
    buffers = arr.buffers()[:2]
    buffers[0] = pa.py_buffer(b'\x05')
    arr = arr.from_buffers(arr.type, len(arr), buffers, children=[arr.values])
    assert arr.to_pylist() == [[1], None, [4, 5, 6]]
    assert arr.offsets.to_pylist() == [0, 1, 3, 6]
    flattened = arr.flatten()
    flattened.validate(full=True)
    assert flattened.type == typ.value_type
    assert flattened.to_pylist() == [1, 4, 5, 6]
    assert arr.values.to_pylist() == [1, 2, 3, 4, 5, 6]