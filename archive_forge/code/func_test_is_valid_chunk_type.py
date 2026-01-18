from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
@pytest.mark.parametrize('arr_type, result', [(WrappedArray, False), (da.Array, False), (EncapsulateNDArray, True), (np.ma.MaskedArray, True), (np.ndarray, True), (float, False), (int, False)])
def test_is_valid_chunk_type(arr_type, result):
    """Test is_valid_chunk_type for correctness"""
    assert is_valid_chunk_type(arr_type) is result