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
@pytest.mark.large_memory
def test_auto_chunking_list_like():
    item = np.ones((2 ** 28,), dtype='uint8')
    data = [item] * (2 ** 3 - 1)
    arr = pa.array(data, type=pa.list_(pa.uint8()))
    assert isinstance(arr, pa.Array)
    assert len(arr) == 7
    item = np.ones((2 ** 28,), dtype='uint8')
    data = [item] * 2 ** 3
    arr = pa.array(data, type=pa.list_(pa.uint8()))
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 2
    assert len(arr.chunk(0)) == 7
    assert len(arr.chunk(1)) == 1
    chunk = arr.chunk(1)
    scalar = chunk[0]
    assert isinstance(scalar, pa.ListScalar)
    expected = pa.array(item, type=pa.uint8())
    assert scalar.values == expected