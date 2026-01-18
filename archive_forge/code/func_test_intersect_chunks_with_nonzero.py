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
def test_intersect_chunks_with_nonzero():
    from dask.array.rechunk import intersect_chunks
    old = ((4, 4), (2,))
    new = ((8,), (1, 1))
    result = list(intersect_chunks(old, new))
    expected = [(((0, slice(0, 4, None)), (0, slice(0, 1, None))), ((1, slice(0, 4, None)), (0, slice(0, 1, None)))), (((0, slice(0, 4, None)), (0, slice(1, 2, None))), ((1, slice(0, 4, None)), (0, slice(1, 2, None))))]
    assert result == expected