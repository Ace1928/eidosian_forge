from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_StackedBytesArray_vectorized_indexing() -> None:
    array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
    stacked = strings.StackedBytesArray(array)
    expected = np.array([[b'abc', b'def'], [b'def', b'abc']])
    V = IndexerMaker(indexing.VectorizedIndexer)
    indexer = V[np.array([[0, 1], [1, 0]])]
    actual = stacked.vindex[indexer]
    assert_array_equal(actual, expected)