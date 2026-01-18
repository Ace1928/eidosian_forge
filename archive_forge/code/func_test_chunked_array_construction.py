from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_construction():
    arr = pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert arr.type == pa.int64()
    assert len(arr) == 9
    assert len(arr.chunks) == 3
    arr = pa.chunked_array([[1, 2, 3], [4.0, 5.0, 6.0], [7, 8, 9]])
    assert arr.type == pa.int64()
    assert len(arr) == 9
    assert len(arr.chunks) == 3
    arr = pa.chunked_array([[1, 2, 3], [4.0, 5.0, 6.0], [7, 8, 9]], type=pa.int8())
    assert arr.type == pa.int8()
    assert len(arr) == 9
    assert len(arr.chunks) == 3
    arr = pa.chunked_array([[1, 2, 3], []])
    assert arr.type == pa.int64()
    assert len(arr) == 3
    assert len(arr.chunks) == 2
    msg = 'cannot construct ChunkedArray from empty vector and omitted type'
    with pytest.raises(ValueError, match=msg):
        assert pa.chunked_array([])
    assert pa.chunked_array([], type=pa.string()).type == pa.string()
    assert pa.chunked_array([[]]).type == pa.null()
    assert pa.chunked_array([[]], type=pa.string()).type == pa.string()