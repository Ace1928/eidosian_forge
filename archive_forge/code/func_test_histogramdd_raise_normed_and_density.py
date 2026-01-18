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
def test_histogramdd_raise_normed_and_density():
    data = da.random.default_rng().random(size=(10, 3), chunks=(5, 3))
    bins = (4, 5, 6)
    ranges = ((0, 1),) * 3
    with pytest.raises(TypeError, match="Cannot specify both 'normed' and 'density'"):
        da.histogramdd(data, bins=bins, range=ranges, normed=True, density=True)