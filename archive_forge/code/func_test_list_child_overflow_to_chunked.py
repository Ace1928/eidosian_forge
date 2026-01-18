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
@pytest.mark.large_memory
def test_list_child_overflow_to_chunked():
    kilobyte_string = 'x' * 1024
    two_mega = 2 ** 21
    vals = [[kilobyte_string]] * (two_mega - 1)
    arr = pa.array(vals)
    assert isinstance(arr, pa.Array)
    assert len(arr) == two_mega - 1
    vals = [[kilobyte_string]] * two_mega
    arr = pa.array(vals)
    assert isinstance(arr, pa.ChunkedArray)
    assert len(arr) == two_mega
    assert len(arr.chunk(0)) == two_mega - 1
    assert len(arr.chunk(1)) == 1