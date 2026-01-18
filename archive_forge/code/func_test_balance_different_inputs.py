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
def test_balance_different_inputs():
    N = 210
    x = da.from_array(np.random.default_rng().uniform(size=(N, N)))
    balanced = x.rechunk(chunks=('10MB', -1), balance=True)
    unbalanced = x.rechunk(chunks=('10MB', -1), balance=False)
    assert balanced.chunks == unbalanced.chunks
    assert balanced.chunks[1] == (N,)