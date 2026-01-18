from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_balance_small():
    arr_len = 13
    x = da.from_array(np.arange(arr_len))
    balanced = x.rechunk(chunks=4, balance=True)
    unbalanced = x.rechunk(chunks=4, balance=False)
    assert balanced.chunks[0] == (5, 5, 3)
    assert unbalanced.chunks[0] == (4, 4, 4, 1)
    arr_len = 7
    x = da.from_array(np.arange(arr_len))
    balanced = x.rechunk(chunks=3, balance=True)
    unbalanced = x.rechunk(chunks=3, balance=False)
    assert balanced.chunks[0] == (4, 3)
    assert unbalanced.chunks[0] == (3, 3, 1)