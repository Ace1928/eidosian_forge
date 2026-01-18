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
def test_c_array_protocol():

    class ArrayWrapper:

        def __init__(self, data):
            self.data = data

        def __arrow_c_array__(self, requested_schema=None):
            return self.data.__arrow_c_array__(requested_schema)
    arr = ArrayWrapper(pa.array([1, 2, 3], type=pa.int64()))
    result = pa.array(arr)
    assert result == arr.data
    result = pa.array(arr, type=pa.int32())
    assert result == pa.array([1, 2, 3], type=pa.int32())