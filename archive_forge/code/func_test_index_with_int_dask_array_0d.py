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
@pytest.mark.parametrize('chunks', [1, 2, 3])
def test_index_with_int_dask_array_0d(chunks):
    x = da.from_array([[10, 20, 30], [40, 50, 60]], chunks=chunks)
    idx0 = da.from_array(1, chunks=1)
    assert_eq(x[idx0, :], x[1, :])
    assert_eq(x[:, idx0], x[:, 1])