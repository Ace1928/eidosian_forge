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
@pytest.mark.parametrize('funcname', ['moveaxis', 'rollaxis'])
@pytest.mark.parametrize('shape', [(), (5,), (3, 5, 7, 3)])
def test_moveaxis_rollaxis(funcname, shape):
    x = np.random.default_rng().random(shape)
    d = da.from_array(x, chunks=len(shape) * (2,))
    np_func = getattr(np, funcname)
    da_func = getattr(da, funcname)
    for axis1 in range(-x.ndim, x.ndim):
        assert isinstance(da_func(d, 0, axis1), da.Array)
        for axis2 in range(-x.ndim, x.ndim):
            assert_eq(np_func(x, axis1, axis2), da_func(d, axis1, axis2))