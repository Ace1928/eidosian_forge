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
def test_intersect_2():
    """Convert 1 D chunks"""
    old = ((20, 20, 20, 20, 20),)
    new = ((58, 4, 20, 18),)
    answer = [(((0, slice(0, 20)),), ((1, slice(0, 20)),), ((2, slice(0, 18)),)), (((2, slice(18, 20)),), ((3, slice(0, 2)),)), (((3, slice(2, 20)),), ((4, slice(0, 2)),)), (((4, slice(2, 20)),),)]
    cross = list(intersect_chunks(old_chunks=old, new_chunks=new))
    assert answer == cross