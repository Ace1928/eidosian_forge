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
def test_binary_array_masked():
    masked_basic = pa.array([b'\x05'], type=pa.binary(1), mask=np.array([False]))
    assert [b'\x05'] == masked_basic.to_pylist()
    masked = pa.array(np.array([b'\x05']), type=pa.binary(1), mask=np.array([False]))
    assert [b'\x05'] == masked.to_pylist()
    masked_nulls = pa.array(np.array([b'\x05']), type=pa.binary(1), mask=np.array([True]))
    assert [None] == masked_nulls.to_pylist()
    masked = pa.array(np.array([b'\x05']), type=pa.binary(), mask=np.array([False]))
    assert [b'\x05'] == masked.to_pylist()
    masked_nulls = pa.array(np.array([b'\x05']), type=pa.binary(), mask=np.array([True]))
    assert [None] == masked_nulls.to_pylist()
    npa = np.array([b'aaa', b'bbb', b'ccc'] * 10)
    arrow_array = pa.array(npa, type=pa.binary(3), mask=np.array([False, False, False] * 10))
    npa[npa == b'bbb'] = b'XXX'
    assert [b'aaa', b'bbb', b'ccc'] * 10 == arrow_array.to_pylist()