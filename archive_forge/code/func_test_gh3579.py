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
def test_gh3579():
    assert_eq(np.arange(10)[0::-1], da.arange(10, chunks=3)[0::-1])
    assert_eq(np.arange(10)[::-1], da.arange(10, chunks=3)[::-1])