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
def test_balance_split_into_n_chunks():
    array_lens = [991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069]
    for N in array_lens:
        for nchunks in range(1, 20):
            x = da.from_array(np.random.default_rng().uniform(size=N))
            y = x.rechunk(chunks=len(x) // nchunks, balance=True)
            assert len(y.chunks[0]) == nchunks