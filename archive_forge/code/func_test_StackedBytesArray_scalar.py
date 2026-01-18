from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_StackedBytesArray_scalar() -> None:
    array = np.array([b'a', b'b', b'c'], dtype='S')
    actual = strings.StackedBytesArray(array)
    expected = np.array(b'abc')
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.size == expected.size
    assert actual.ndim == expected.ndim
    with pytest.raises(TypeError):
        len(actual)
    np.testing.assert_array_equal(expected, actual)
    B = IndexerMaker(indexing.BasicIndexer)
    with pytest.raises(IndexError):
        actual[B[:2]]