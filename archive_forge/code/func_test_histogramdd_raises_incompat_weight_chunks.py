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
def test_histogramdd_raises_incompat_weight_chunks():
    rng = da.random.default_rng()
    x = rng.random(size=(10,), chunks=2)
    y = rng.random(size=(10,), chunks=2)
    z = da.atleast_2d((x, y)).T.rechunk((2, 2))
    w = rng.random(size=(10,), chunks=5)
    with pytest.raises(ValueError, match='Input arrays and weights must have the same shape and chunk structure.'):
        da.histogramdd((x, y), bins=(3,) * 2, range=((0, 1),) * 2, weights=w)
    with pytest.raises(ValueError, match='Input array and weights must have the same shape and chunk structure along the first dimension.'):
        da.histogramdd(z, bins=(3,) * 2, range=((0, 1),) * 2, weights=w)