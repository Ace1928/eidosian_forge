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
def test_map_array_values_offsets():
    ty = pa.map_(pa.utf8(), pa.int32())
    ty_values = pa.struct([pa.field('key', pa.utf8(), nullable=False), pa.field('value', pa.int32())])
    a = pa.array([[('a', 1), ('b', 2)], [('c', 3)]], type=ty)
    assert a.values.type.equals(ty_values)
    assert a.values == pa.array([{'key': 'a', 'value': 1}, {'key': 'b', 'value': 2}, {'key': 'c', 'value': 3}], type=ty_values)
    assert a.keys.equals(pa.array(['a', 'b', 'c']))
    assert a.items.equals(pa.array([1, 2, 3], type=pa.int32()))
    assert pa.ListArray.from_arrays(a.offsets, a.keys).equals(pa.array([['a', 'b'], ['c']]))
    assert pa.ListArray.from_arrays(a.offsets, a.items).equals(pa.array([[1, 2], [3]], type=pa.list_(pa.int32())))
    with pytest.raises(NotImplementedError):
        a.flatten()