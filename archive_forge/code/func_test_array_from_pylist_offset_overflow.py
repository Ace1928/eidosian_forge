import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.slow
@pytest.mark.large_memory
def test_array_from_pylist_offset_overflow():
    items = [b'a'] * 2 ** 31
    arr = pa.array(items, type=pa.string())
    assert isinstance(arr, pa.ChunkedArray)
    assert len(arr) == 2 ** 31
    assert len(arr.chunks) > 1
    mask = np.zeros(2 ** 31, bool)
    arr = pa.array(items, mask=mask, type=pa.string())
    assert isinstance(arr, pa.ChunkedArray)
    assert len(arr) == 2 ** 31
    assert len(arr.chunks) > 1
    arr = pa.array(items, type=pa.binary())
    assert isinstance(arr, pa.ChunkedArray)
    assert len(arr) == 2 ** 31
    assert len(arr.chunks) > 1