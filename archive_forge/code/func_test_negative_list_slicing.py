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
def test_negative_list_slicing():
    x = np.arange(5)
    dx = da.from_array(x, chunks=2)
    assert_eq(dx[[0, -5]], x[[0, -5]])
    assert_eq(dx[[4, -1]], x[[4, -1]])