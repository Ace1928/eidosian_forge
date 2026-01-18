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
def test_cast_integers_unsafe():
    unsafe_cases = [(np.array([50000], dtype='i4'), 'int32', np.array([50000]).astype(dtype='i2'), pa.int16()), (np.array([70000], dtype='i4'), 'int32', np.array([70000]).astype(dtype='u2'), pa.uint16()), (np.array([-1], dtype='i4'), 'int32', np.array([-1]).astype(dtype='u2'), pa.uint16()), (np.array([50000], dtype='u2'), pa.uint16(), np.array([50000]).astype(dtype='i2'), pa.int16())]
    for case in unsafe_cases:
        _check_cast_case(case, safe=False)