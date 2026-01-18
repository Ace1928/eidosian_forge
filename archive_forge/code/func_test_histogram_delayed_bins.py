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
def test_histogram_delayed_bins(density, weighted):
    n = 100
    v = np.random.default_rng().random(n)
    bins = np.array([0, 0.2, 0.5, 0.8, 1])
    vd = da.from_array(v, chunks=10)
    bins_d = da.from_array(bins, chunks=2)
    if weighted:
        weights = np.random.default_rng().random(n)
        weights_d = da.from_array(weights, chunks=vd.chunks)
    hist_d, bins_d2 = da.histogram(vd, bins=bins_d, range=[bins_d[0], bins_d[-1]], density=density, weights=weights_d if weighted else None)
    hist, bins = np.histogram(v, bins=bins, range=[bins[0], bins[-1]], density=density, weights=weights if weighted else None)
    assert bins_d is bins_d2
    assert_eq(hist_d, hist)
    assert_eq(bins_d2, bins)