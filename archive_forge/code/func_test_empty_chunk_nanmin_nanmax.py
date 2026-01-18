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
@pytest.mark.parametrize('func', ['nanmin', 'nanmax'])
def test_empty_chunk_nanmin_nanmax(func):
    x = np.arange(10).reshape(2, 5)
    d = da.from_array(x, chunks=2)
    x = x[x > 4]
    d = d[d > 4]
    block_lens = np.array([len(x.compute()) for x in d.blocks])
    assert 0 in block_lens
    with pytest.raises(ValueError) as err:
        getattr(da, func)(d)
    assert 'Arrays chunk sizes are unknown' in str(err)
    d = d.compute_chunk_sizes()
    assert_eq(getattr(da, func)(d), getattr(np, func)(x))