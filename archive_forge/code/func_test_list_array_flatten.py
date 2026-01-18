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
@pytest.mark.parametrize(('offset_type', 'list_type_factory'), [(pa.int32(), pa.list_), (pa.int64(), pa.large_list)])
def test_list_array_flatten(offset_type, list_type_factory):
    typ2 = list_type_factory(list_type_factory(pa.int64()))
    arr2 = pa.array([None, [[1, None, 2], None, [3, 4]], [], [[], [5, 6], None], [[7, 8]]], type=typ2)
    offsets2 = pa.array([0, 0, 3, 3, 6, 7], type=offset_type)
    typ1 = list_type_factory(pa.int64())
    arr1 = pa.array([[1, None, 2], None, [3, 4], [], [5, 6], None, [7, 8]], type=typ1)
    offsets1 = pa.array([0, 3, 3, 5, 5, 7, 7, 9], type=offset_type)
    arr0 = pa.array([1, None, 2, 3, 4, 5, 6, 7, 8], type=pa.int64())
    assert arr2.flatten().equals(arr1)
    assert arr2.offsets.equals(offsets2)
    assert arr2.values.equals(arr1)
    assert arr1.flatten().equals(arr0)
    assert arr1.offsets.equals(offsets1)
    assert arr1.values.equals(arr0)
    assert arr2.flatten().flatten().equals(arr0)
    assert arr2.values.values.equals(arr0)