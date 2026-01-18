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
def test_array_supported_masks():
    arr = pa.array([4, None, 4, 3.0], mask=np.array([False, True, False, True]))
    assert arr.to_pylist() == [4, None, 4, None]
    arr = pa.array([4, None, 4, 3], mask=pa.array([False, True, False, True]))
    assert arr.to_pylist() == [4, None, 4, None]
    arr = pa.array([4, None, 4, 3], mask=[False, True, False, True])
    assert arr.to_pylist() == [4, None, 4, None]
    arr = pa.array([4, 3, None, 3], mask=[False, True, False, True])
    assert arr.to_pylist() == [4, None, None, None]
    with pytest.raises(pa.ArrowTypeError):
        arr = pa.array([4, None, 4, 3], mask=pa.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(pa.ArrowTypeError):
        arr = pa.array([4, None, 4, 3], mask=[1.0, 2.0, 3.0, 4.0])
    with pytest.raises(pa.ArrowTypeError):
        arr = pa.array([4, None, 4, 3], mask=np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(pa.ArrowTypeError):
        arr = pa.array([4, None, 4, 3], mask=pa.array([False, True, False, True], mask=pa.array([True, True, True, True])))
    with pytest.raises(pa.ArrowTypeError):
        arr = pa.array([4, None, 4, 3], mask=pa.array([False, None, False, True]))
    with pytest.raises(TypeError):
        arr = pa.array(np.array([4, None, 4, 3.0]), mask=[True, False, True, False])
    with pytest.raises(TypeError):
        arr = pa.array(np.array([4, None, 4, 3.0]), mask=pa.array([True, False, True, False]))