from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_equals():

    def eq(xarrs, yarrs):
        if isinstance(xarrs, pa.ChunkedArray):
            x = xarrs
        else:
            x = pa.chunked_array(xarrs)
        if isinstance(yarrs, pa.ChunkedArray):
            y = yarrs
        else:
            y = pa.chunked_array(yarrs)
        assert x.equals(y)
        assert y.equals(x)
        assert x == y
        assert x != str(y)

    def ne(xarrs, yarrs):
        if isinstance(xarrs, pa.ChunkedArray):
            x = xarrs
        else:
            x = pa.chunked_array(xarrs)
        if isinstance(yarrs, pa.ChunkedArray):
            y = yarrs
        else:
            y = pa.chunked_array(yarrs)
        assert not x.equals(y)
        assert not y.equals(x)
        assert x != y
    eq(pa.chunked_array([], type=pa.int32()), pa.chunked_array([], type=pa.int32()))
    ne(pa.chunked_array([], type=pa.int32()), pa.chunked_array([], type=pa.int64()))
    a = pa.array([0, 2], type=pa.int32())
    b = pa.array([0, 2], type=pa.int64())
    c = pa.array([0, 3], type=pa.int32())
    d = pa.array([0, 2, 0, 3], type=pa.int32())
    eq([a], [a])
    ne([a], [b])
    eq([a, c], [a, c])
    eq([a, c], [d])
    ne([c, a], [a, c])
    assert not pa.chunked_array([], type=pa.int32()).equals(None)