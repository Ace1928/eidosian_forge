from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('axis', [None, 0, (1,), (0, 1)])
def test_count_nonzero_axis(axis):
    for shape, chunks in [((0, 0), (0, 0)), ((15, 16), (4, 5))]:
        x = np.random.default_rng().integers(10, size=shape)
        d = da.from_array(x, chunks=chunks)
        x_c = np.count_nonzero(x, axis)
        d_c = da.count_nonzero(d, axis)
        if d_c.shape == tuple():
            assert x_c == d_c.compute()
        else:
            assert_eq(x_c, d_c)