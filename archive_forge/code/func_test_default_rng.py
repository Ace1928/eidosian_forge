from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
@pytest.mark.parametrize('sd', [None, 42, np.random.PCG64, da.random.Generator(np.random.PCG64)], ids=type)
def test_default_rng(sd):
    rng = da.random.default_rng(seed=sd)
    assert isinstance(rng, da.random.Generator)