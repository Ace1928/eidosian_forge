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
@pytest.mark.parametrize('chunks', [1, 2, 3, 4, 5])
def test_index_with_int_dask_array_nanchunks(chunks):
    a = da.arange(-2, 3, chunks=chunks)
    assert_eq(a[a.nonzero()], np.array([-2, -1, 1, 2]))
    a = da.zeros(5, chunks=chunks)
    assert_eq(a[a.nonzero()], np.array([]))