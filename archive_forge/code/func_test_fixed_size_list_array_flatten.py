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
def test_fixed_size_list_array_flatten():
    typ2 = pa.list_(pa.list_(pa.int64(), 2), 3)
    arr2 = pa.array([[[1, 2], [3, 4], [5, 6]], None, [[7, None], None, [8, 9]]], type=typ2)
    assert arr2.type.equals(typ2)
    typ1 = pa.list_(pa.int64(), 2)
    arr1 = pa.array([[1, 2], [3, 4], [5, 6], [7, None], None, [8, 9]], type=typ1)
    assert arr1.type.equals(typ1)
    assert arr2.flatten().equals(arr1)
    typ0 = pa.int64()
    arr0 = pa.array([1, 2, 3, 4, 5, 6, 7, None, 8, 9], type=typ0)
    assert arr0.type.equals(typ0)
    assert arr1.flatten().equals(arr0)
    assert arr2.flatten().flatten().equals(arr0)