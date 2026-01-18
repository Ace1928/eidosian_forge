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
@pytest.mark.parametrize('density', [True, False])
@pytest.mark.parametrize('weighted', [True, False])
@pytest.mark.parametrize('non_delayed_i', [None, 0, 1])
@pytest.mark.parametrize('delay_n_bins', [False, True])
def test_histogram_delayed_range(density, weighted, non_delayed_i, delay_n_bins):
    n = 100
    v = np.random.default_rng().random(n)
    vd = da.from_array(v, chunks=10)
    if weighted:
        weights = np.random.default_rng().random(n)
        weights_d = da.from_array(weights, chunks=vd.chunks)
    d_range = [vd.min(), vd.max()]
    if non_delayed_i is not None:
        d_range[non_delayed_i] = d_range[non_delayed_i].compute()
    hist_d, bins_d = da.histogram(vd, bins=da.array(n) if delay_n_bins and (not density) else n, range=d_range, density=density, weights=weights_d if weighted else None)
    hist, bins = np.histogram(v, bins=n, range=[v.min(), v.max()], density=density, weights=weights if weighted else None)
    assert_eq(hist_d, hist)
    assert_eq(bins_d, bins)