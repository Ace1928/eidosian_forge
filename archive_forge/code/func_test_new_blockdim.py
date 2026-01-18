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
def test_new_blockdim():
    assert new_blockdim(20, [5, 5, 5, 5], slice(0, None, 2)) == [3, 2, 3, 2]