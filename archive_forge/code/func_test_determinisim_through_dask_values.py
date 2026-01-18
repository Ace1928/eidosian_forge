from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_determinisim_through_dask_values(generator_class):
    samples_1 = generator_class(42).normal(size=1000, chunks=10)
    samples_2 = generator_class(42).normal(size=1000, chunks=10)
    assert set(samples_1.dask) == set(samples_2.dask)
    assert_eq(samples_1, samples_2)