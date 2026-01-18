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
def test_histogramdd_raise_incompat_shape():
    data = da.random.default_rng().random(size=(10,), chunks=(2,))
    with pytest.raises(ValueError, match='Single array input to histogramdd should be columnar'):
        da.histogramdd(data, bins=4, range=((-3, 3),))
    data = da.random.default_rng().random(size=(4, 4, 4), chunks=(2, 2, 2))
    with pytest.raises(ValueError, match='Single array input to histogramdd should be columnar'):
        da.histogramdd(data, bins=4, range=((-3, 3),))