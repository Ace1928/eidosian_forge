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
def test_map_from_arrays():
    offsets_arr = np.array([0, 2, 5, 8], dtype='i4')
    offsets = pa.array(offsets_arr, type='int32')
    pykeys = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
    pyitems = list(range(len(pykeys)))
    pypairs = list(zip(pykeys, pyitems))
    pyentries = [pypairs[:2], pypairs[2:5], pypairs[5:8]]
    keys = pa.array(pykeys, type='binary')
    items = pa.array(pyitems, type='i4')
    result = pa.MapArray.from_arrays(offsets, keys, items)
    expected = pa.array(pyentries, type=pa.map_(pa.binary(), pa.int32()))
    assert result.equals(expected)
    offsets = [0, None, 2, 6]
    pykeys = [b'a', b'b', b'c', b'd', b'e', b'f']
    pyitems = [1, 2, 3, None, 4, 5]
    pypairs = list(zip(pykeys, pyitems))
    pyentries = [pypairs[:2], None, pypairs[2:]]
    keys = pa.array(pykeys, type='binary')
    items = pa.array(pyitems, type='i4')
    result = pa.MapArray.from_arrays(offsets, keys, items)
    expected = pa.array(pyentries, type=pa.map_(pa.binary(), pa.int32()))
    assert result.equals(expected)
    result = pa.MapArray.from_arrays(offsets, keys, items, pa.map_(keys.type, items.type))
    assert result.equals(expected)
    with pytest.raises(pa.ArrowTypeError, match='Expected map type, got string'):
        pa.MapArray.from_arrays(offsets, keys, items, pa.string())
    with pytest.raises(pa.ArrowTypeError, match='Mismatching map items type'):
        pa.MapArray.from_arrays(offsets, keys, items, pa.map_(keys.type, pa.int64()))
    offsets = [0, 1, 3, 5]
    keys = np.arange(5)
    items = np.arange(5)
    _ = pa.MapArray.from_arrays(offsets, keys, items)
    with pytest.raises(ValueError):
        pa.MapArray.from_arrays(offsets + [6], keys, items)
    with pytest.raises(ValueError):
        pa.MapArray.from_arrays(offsets, keys, np.concatenate([items, items]))
    keys_with_null = list(keys)[:-1] + [None]
    assert len(keys_with_null) == len(items)
    with pytest.raises(ValueError):
        pa.MapArray.from_arrays(offsets, keys_with_null, items)