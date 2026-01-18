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
@pytest.mark.parametrize('klass', [pa.ListArray, pa.LargeListArray])
def test_list_array_values_offsets_sliced(klass):
    arr = klass.from_arrays(offsets=[0, 3, 4, 6], values=[1, 2, 3, 4, 5, 6])
    assert arr.values.to_pylist() == [1, 2, 3, 4, 5, 6]
    assert arr.offsets.to_pylist() == [0, 3, 4, 6]
    arr2 = arr[1:]
    assert arr2.values.to_pylist() == [1, 2, 3, 4, 5, 6]
    assert arr2.offsets.to_pylist() == [3, 4, 6]
    assert arr2.flatten().to_pylist() == [4, 5, 6]
    i = arr2.offsets[0].as_py()
    j = arr2.offsets[1].as_py()
    assert arr2[0].as_py() == arr2.values[i:j].to_pylist() == [4]