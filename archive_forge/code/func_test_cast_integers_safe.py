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
def test_cast_integers_safe():
    safe_cases = [(np.array([0, 1, 2, 3], dtype='i1'), 'int8', np.array([0, 1, 2, 3], dtype='i4'), pa.int32()), (np.array([0, 1, 2, 3], dtype='i1'), 'int8', np.array([0, 1, 2, 3], dtype='u4'), pa.uint16()), (np.array([0, 1, 2, 3], dtype='i1'), 'int8', np.array([0, 1, 2, 3], dtype='u1'), pa.uint8()), (np.array([0, 1, 2, 3], dtype='i1'), 'int8', np.array([0, 1, 2, 3], dtype='f8'), pa.float64())]
    for case in safe_cases:
        _check_cast_case(case)
    unsafe_cases = [(np.array([50000], dtype='i4'), 'int32', 'int16'), (np.array([70000], dtype='i4'), 'int32', 'uint16'), (np.array([-1], dtype='i4'), 'int32', 'uint16'), (np.array([50000], dtype='u2'), 'uint16', 'int16')]
    for in_data, in_type, out_type in unsafe_cases:
        in_arr = pa.array(in_data, type=in_type)
        with pytest.raises(pa.ArrowInvalid):
            in_arr.cast(out_type)