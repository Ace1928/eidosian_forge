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
def test_histogramdd_raises_incompat_bins_or_range():
    data = da.random.default_rng().random(size=(10, 4), chunks=(5, 4))
    bins = (2, 3, 4, 5)
    ranges = ((0, 1),) * len(bins)
    bins = (2, 3, 4)
    with pytest.raises(ValueError, match='The dimension of bins must be equal to the dimension of the sample.'):
        da.histogramdd(data, bins=bins, range=ranges)
    bins = (2, 3, 4, 5)
    ranges = ((0, 1),) * 3
    with pytest.raises(ValueError, match='range argument requires one entry, a min max pair, per dimension.'):
        da.histogramdd(data, bins=bins, range=ranges)
    with pytest.raises(ValueError, match='range argument should be a sequence of pairs'):
        da.histogramdd(data, bins=bins, range=((0, 1), (0, 1, 2), 3, 5))