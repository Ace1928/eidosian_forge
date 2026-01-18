from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_StackedBytesArray() -> None:
    array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
    actual = strings.StackedBytesArray(array)
    expected = np.array([b'abc', b'def'], dtype='S')
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.size == expected.size
    assert actual.ndim == expected.ndim
    assert len(actual) == len(expected)
    assert_array_equal(expected, actual)
    B = IndexerMaker(indexing.BasicIndexer)
    assert_array_equal(expected[:1], actual[B[:1]])
    with pytest.raises(IndexError):
        actual[B[:, :2]]