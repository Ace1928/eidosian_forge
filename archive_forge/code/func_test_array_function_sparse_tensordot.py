from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
def test_array_function_sparse_tensordot():
    sparse = pytest.importorskip('sparse')
    rng = np.random.default_rng()
    x = rng.random((2, 3, 4))
    x[x < 0.9] = 0
    y = rng.random((4, 3, 2))
    y[y < 0.9] = 0
    xx = sparse.COO(x)
    yy = sparse.COO(y)
    assert_eq(np.tensordot(x, y, axes=(2, 0)), np.tensordot(xx, yy, axes=(2, 0)).todense())