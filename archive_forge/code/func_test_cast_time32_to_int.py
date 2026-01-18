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
def test_cast_time32_to_int():
    arr = pa.array(np.array([0, 1, 2], dtype='int32'), type=pa.time32('s'))
    expected = pa.array([0, 1, 2], type='i4')
    result = arr.cast('i4')
    assert result.equals(expected)