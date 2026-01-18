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
def test_auto_chunking_binary_like():
    v1 = b'x' * 100000000
    v2 = b'x' * 147483646
    one_chunk_data = [v1] * 20 + [b'', None, v2]
    arr = pa.array(one_chunk_data, type=pa.binary())
    assert isinstance(arr, pa.Array)
    assert len(arr) == 23
    assert arr[20].as_py() == b''
    assert arr[21].as_py() is None
    assert arr[22].as_py() == v2
    two_chunk_data = one_chunk_data + [b'two']
    arr = pa.array(two_chunk_data, type=pa.binary())
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 2
    assert len(arr.chunk(0)) == 23
    assert len(arr.chunk(1)) == 1
    assert arr.chunk(0)[20].as_py() == b''
    assert arr.chunk(0)[21].as_py() is None
    assert arr.chunk(0)[22].as_py() == v2
    assert arr.chunk(1).to_pylist() == [b'two']
    three_chunk_data = one_chunk_data * 2 + [b'three', b'three']
    arr = pa.array(three_chunk_data, type=pa.binary())
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 3
    assert len(arr.chunk(0)) == 23
    assert len(arr.chunk(1)) == 23
    assert len(arr.chunk(2)) == 2
    for i in range(2):
        assert arr.chunk(i)[20].as_py() == b''
        assert arr.chunk(i)[21].as_py() is None
        assert arr.chunk(i)[22].as_py() == v2
    assert arr.chunk(2).to_pylist() == [b'three', b'three']