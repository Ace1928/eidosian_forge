from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
import dask
import dask.array as da
from dask.array.reductions import nannumel, numel
from dask.array.utils import assert_eq
def test_from_delayed_meta():

    def f():
        return sparse.COO.from_numpy(np.eye(3))
    d = dask.delayed(f)()
    x = da.from_delayed(d, shape=(3, 3), meta=sparse.COO.from_numpy(np.eye(1)))
    assert isinstance(x._meta, sparse.COO)
    assert_eq(x, x)