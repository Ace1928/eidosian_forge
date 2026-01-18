from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
@pytest.mark.parametrize('func', ['nansum', 'sum', 'nanmin', 'min', 'nanmax', 'max'])
def test_nan_object(func):
    with warnings.catch_warnings():
        if os.name == 'nt' and func in {'min', 'max'}:
            warnings.simplefilter('ignore', RuntimeWarning)
        x = np.array([[1, np.nan, 3, 4], [5, 6, 7, np.nan], [9, 10, 11, 12]]).astype(object)
        d = da.from_array(x, chunks=(2, 2))
        if func in {'nanmin', 'nanmax'}:
            warnings.simplefilter('ignore', RuntimeWarning)
        assert_eq(getattr(np, func)(x, axis=()), getattr(da, func)(d, axis=()))
        if func in {'nanmin', 'nanmax'}:
            warnings.simplefilter('default', RuntimeWarning)
        if func in {'min', 'max'}:
            warnings.simplefilter('ignore', RuntimeWarning)
        assert_eq(getattr(np, func)(x, axis=0), getattr(da, func)(d, axis=0))
        if os.name != 'nt' and func in {'min', 'max'}:
            warnings.simplefilter('default', RuntimeWarning)
        assert_eq(getattr(np, func)(x, axis=1), getattr(da, func)(d, axis=1))
        assert_eq(np.array(getattr(np, func)(x)).astype(object), getattr(da, func)(d))