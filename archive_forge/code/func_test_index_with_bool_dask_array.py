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
def test_index_with_bool_dask_array():
    x = np.arange(36).reshape((6, 6))
    d = da.from_array(x, chunks=(3, 3))
    ind = np.asarray([True, True, False, True, False, False], dtype=bool)
    ind = da.from_array(ind, chunks=2)
    for index in [ind, (slice(1, 9, 2), ind), (ind, slice(2, 8, 1))]:
        x_index = dask.compute(index)[0]
        assert_eq(x[x_index], d[index])