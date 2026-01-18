from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
def test_cupy_unsupported():
    with config.set({'array.backend': 'cupy'}):
        x = da.arange(12, chunks=3)
        da.random.default_rng(np.random.PCG64()).permutation(x).compute()
        with pytest.raises(NotImplementedError):
            da.random.default_rng().permutation(x).compute()
        with pytest.raises(NotImplementedError):
            da.random.default_rng().choice(10).compute()