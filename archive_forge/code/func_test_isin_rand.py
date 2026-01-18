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
@pytest.mark.parametrize('seed', [23, 796])
@pytest.mark.parametrize('low, high', [[0, 10]])
@pytest.mark.parametrize('elements_shape, elements_chunks', [[(10,), (5,)], [(10,), (3,)], [(4, 5), (3, 2)], [(20, 20), (4, 5)]])
@pytest.mark.parametrize('test_shape, test_chunks', [[(10,), (5,)], [(10,), (3,)], [(4, 5), (3, 2)], [(20, 20), (4, 5)]])
@pytest.mark.parametrize('invert', [True, False])
def test_isin_rand(seed, low, high, elements_shape, elements_chunks, test_shape, test_chunks, invert):
    rng = np.random.default_rng(seed)
    a1 = rng.integers(low, high, size=elements_shape)
    d1 = da.from_array(a1, chunks=elements_chunks)
    a2 = rng.integers(low, high, size=test_shape) - 5
    d2 = da.from_array(a2, chunks=test_chunks)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=da.PerformanceWarning)
        r_a = np.isin(a1, a2, invert=invert)
        r_d = da.isin(d1, d2, invert=invert)
    assert_eq(r_a, r_d)