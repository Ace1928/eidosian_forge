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
def test_array_slice_negative_step():
    np_arr = np.arange(20)
    arr = pa.array(np_arr)
    chunked_arr = pa.chunked_array([arr])
    cases = [slice(None, None, -1), slice(None, 6, -2), slice(10, 6, -2), slice(8, None, -2), slice(2, 10, -2), slice(10, 2, -2), slice(None, None, 2), slice(0, 10, 2)]
    for case in cases:
        result = arr[case]
        expected = pa.array(np_arr[case])
        assert result.equals(expected)
        result = pa.record_batch([arr], names=['f0'])[case]
        expected = pa.record_batch([expected], names=['f0'])
        assert result.equals(expected)
        result = chunked_arr[case]
        expected = pa.chunked_array([np_arr[case]])
        assert result.equals(expected)