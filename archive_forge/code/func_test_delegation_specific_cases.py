from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
def test_delegation_specific_cases():
    a = da.from_array(['a', 'b', '.', 'd'])
    assert_eq(a == '.', [False, False, True, False])
    assert_eq('.' == a, [False, False, True, False])
    assert 'b' in a