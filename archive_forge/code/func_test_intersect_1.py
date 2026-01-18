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
def test_intersect_1():
    """Convert 1 D chunks"""
    old = ((10, 10, 10, 10, 10),)
    new = ((25, 5, 20),)
    answer = [(((0, slice(0, 10)),), ((1, slice(0, 10)),), ((2, slice(0, 5)),)), (((2, slice(5, 10)),),), (((3, slice(0, 10)),), ((4, slice(0, 10)),))]
    cross = list(intersect_chunks(old_chunks=old, new_chunks=new))
    assert answer == cross