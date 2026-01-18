from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
def test_indices_dimensions_chunks():
    chunks = ((1, 4, 2, 3), (5, 5))
    darr = da.indices((10, 10), chunks=chunks)
    assert darr.chunks == ((1, 1),) + chunks
    with dask.config.set({'array.chunk-size': '50 MiB'}):
        shape = (10000, 10000)
        expected = normalize_chunks('auto', shape=shape, dtype=int)
        result = da.indices(shape, chunks='auto')
        actual = result.chunks[1:]
        assert expected == actual