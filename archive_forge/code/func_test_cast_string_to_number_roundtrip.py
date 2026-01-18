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
def test_cast_string_to_number_roundtrip():
    cases = [(pa.array(['1', '127', '-128']), pa.array([1, 127, -128], type=pa.int8())), (pa.array([None, '18446744073709551615']), pa.array([None, 18446744073709551615], type=pa.uint64()))]
    for in_arr, expected in cases:
        casted = in_arr.cast(expected.type, safe=True)
        casted.validate(full=True)
        assert casted.equals(expected)
        casted_back = casted.cast(in_arr.type, safe=True)
        casted_back.validate(full=True)
        assert casted_back.equals(in_arr)