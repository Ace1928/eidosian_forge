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
def test_getitem_avoids_large_chunks():
    with dask.config.set({'array.chunk-size': '0.1Mb'}):
        a = np.arange(2 * 128 * 128, dtype='int64').reshape(2, 128, 128)
        indexer = [0] + [1] * 11
        arr = da.from_array(a, chunks=(1, 8, 8))
        result = arr[indexer]
        expected = a[indexer]
        assert_eq(result, expected)
        arr = da.from_array(a, chunks=(1, 128, 128))
        expected = a[indexer]
        with pytest.warns(da.PerformanceWarning):
            result = arr[indexer]
        assert_eq(result, expected)
        assert result.chunks == ((1, 11), (128,), (128,))
        with dask.config.set({'array.slicing.split-large-chunks': False}):
            with warnings.catch_warnings(record=True) as record:
                result = arr[indexer]
            assert_eq(result, expected)
            assert not record
        with dask.config.set({'array.slicing.split-large-chunks': True}):
            with warnings.catch_warnings(record=True) as record:
                result = arr[indexer]
            assert_eq(result, expected)
            assert not record
            assert result.chunks == ((1,) * 12, (128,), (128,))