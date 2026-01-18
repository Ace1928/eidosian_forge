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
def test_None_overlap_int():
    a, b, c, d = (0, slice(None, 2, None), None, Ellipsis)
    shape = (2, 3, 5, 7, 11)
    x = np.arange(np.prod(shape)).reshape(shape)
    y = da.core.asarray(x)
    xx = x[a, b, c, d]
    yy = y[a, b, c, d]
    assert_eq(xx, yy)