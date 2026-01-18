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
def test_array_getitem():
    arr = pa.array(range(10, 15))
    lst = arr.to_pylist()
    for idx in range(-len(arr), len(arr)):
        assert arr[idx].as_py() == lst[idx]
    for idx in range(-2 * len(arr), -len(arr)):
        with pytest.raises(IndexError):
            arr[idx]
    for idx in range(len(arr), 2 * len(arr)):
        with pytest.raises(IndexError):
            arr[idx]
    for idx in range(-len(arr), len(arr)):
        assert arr[np.int32(idx)].as_py() == lst[idx]