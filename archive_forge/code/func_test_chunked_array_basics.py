from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_basics():
    data = pa.chunked_array([], type=pa.string())
    assert data.type == pa.string()
    assert data.to_pylist() == []
    data.validate()
    data2 = pa.chunked_array([], type='binary')
    assert data2.type == pa.binary()
    with pytest.raises(ValueError):
        pa.chunked_array([])
    data = pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert isinstance(data.chunks, list)
    assert all((isinstance(c, pa.lib.Int64Array) for c in data.chunks))
    assert all((isinstance(c, pa.lib.Int64Array) for c in data.iterchunks()))
    assert len(data.chunks) == 3
    assert data.get_total_buffer_size() == sum((c.get_total_buffer_size() for c in data.iterchunks()))
    assert sys.getsizeof(data) >= object.__sizeof__(data) + data.get_total_buffer_size()
    assert data.nbytes == 3 * 3 * 8
    data.validate()
    wr = weakref.ref(data)
    assert wr() is not None
    del data
    assert wr() is None