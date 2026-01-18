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
@pytest.mark.parametrize('chunks', [(x,) * 3 for x in range(16, 32)])
def test_coarsen_bad_chunks(chunks):
    x1 = da.arange(np.sum(chunks), chunks=5)
    x2 = x1.rechunk(tuple(chunks))
    assert_eq(da.coarsen(np.sum, x1, {0: 10}, trim_excess=True), da.coarsen(np.sum, x2, {0: 10}, trim_excess=True))