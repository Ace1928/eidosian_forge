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
def test_array_string_from_all_null():
    vals = np.array([None, None], dtype=object)
    arr = pa.array(vals, type=pa.string())
    assert arr.null_count == 2
    vals = np.array([np.nan, np.nan], dtype='float64')
    with pytest.raises(TypeError):
        pa.array(vals, type=pa.string())
    arr = pa.array(vals, type=pa.string(), from_pandas=True)
    assert arr.null_count == 2