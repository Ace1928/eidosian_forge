from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [lambda x: np.real(x), lambda x: np.imag(x), lambda x: np.transpose(x)])
def test_array_function_sparse(func):
    sparse = pytest.importorskip('sparse')
    x = da.random.default_rng().random((500, 500), chunks=(100, 100))
    x[x < 0.9] = 0
    y = x.map_blocks(sparse.COO)
    assert_eq(func(x), func(y))