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
@pytest.mark.parametrize('chunks', [2, 4])
def test_index_with_int_dask_array_negindex(chunks):
    a = da.arange(4, chunks=chunks)
    idx = da.from_array([-1, -4], chunks=1)
    assert_eq(a[idx], np.array([3, 0]))