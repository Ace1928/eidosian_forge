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
@pytest.mark.parametrize('funcname', ['atleast_1d', 'atleast_2d', 'atleast_3d'])
@pytest.mark.parametrize('shape, chunks', [(tuple(), tuple()), ((4,), (2,)), ((4, 6), (2, 3)), ((4, 6, 8), (2, 3, 4)), ((4, 6, 8, 10), (2, 3, 4, 5))])
def test_atleast_nd_one_arg(funcname, shape, chunks):
    np_a = np.random.default_rng().random(shape)
    da_a = da.from_array(np_a, chunks=chunks)
    np_func = getattr(np, funcname)
    da_func = getattr(da, funcname)
    np_r = np_func(np_a)
    da_r = da_func(da_a)
    assert_eq(np_r, da_r)