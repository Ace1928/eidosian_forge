from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_getitem_avoids_large_chunks_missing():
    with dask.config.set({'array.chunk-size': '0.1Mb'}):
        a = np.arange(4 * 500 * 500).reshape(4, 500, 500)
        arr = da.from_array(a, chunks=(1, 500, 500))
        arr._chunks = ((1, 1, 1, 1), (np.nan,), (np.nan,))
        indexer = [0, 1] + [2] * 100 + [3]
        expected = a[indexer]
        result = arr[indexer]
        assert_eq(result, expected)