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
@pytest.mark.parametrize('weights', [True, False])
@pytest.mark.parametrize('density', [True, False])
def test_histogram2d_array_bins(weights, density):
    rng = da.random.default_rng()
    n = 800
    xbins = [0.0, 0.2, 0.6, 0.9, 1.0]
    ybins = [0.0, 0.1, 0.4, 0.5, 1.0]
    b = [xbins, ybins]
    x = rng.uniform(0, 1, size=(n,), chunks=(200,))
    y = rng.uniform(0, 1, size=(n,), chunks=(200,))
    w = rng.uniform(0.2, 1.1, size=(n,), chunks=(200,)) if weights else None
    a1, b1x, b1y = da.histogram2d(x, y, bins=b, density=density, weights=w)
    a2, b2x, b2y = np.histogram2d(x, y, bins=b, density=density, weights=w)
    a3, b3x, b3y = np.histogram2d(x.compute(), y.compute(), bins=b, density=density, weights=w.compute() if weights else None)
    assert_eq(a1, a2)
    assert_eq(a1, a3)
    if not (weights or density):
        assert a1.sum() == n
        assert a2.sum() == n
    assert same_keys(da.histogram2d(x, y, bins=b, density=density, weights=w)[0], a1)
    assert a1.compute().shape == a3.shape