from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_RandomState_only_funcs():
    da.random.randint(10, size=5, chunks=3).compute()
    with pytest.warns(DeprecationWarning):
        da.random.random_integers(10, size=5, chunks=3).compute()
    da.random.random_sample(10, chunks=3).compute()