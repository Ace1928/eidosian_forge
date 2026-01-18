from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_generator_consistent_names(generator_class):
    state1 = generator_class(42)
    state2 = generator_class(42)
    assert sorted(state1.normal(size=(100, 100), chunks=(10, 10)).dask) == sorted(state2.normal(size=(100, 100), chunks=(10, 10)).dask)
    assert sorted(state1.normal(size=100, loc=4.5, scale=5.0, chunks=10).dask) == sorted(state2.normal(size=100, loc=4.5, scale=5.0, chunks=10).dask)