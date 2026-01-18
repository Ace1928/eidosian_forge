from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [lambda x: np.min_scalar_type(x), lambda x: np.linalg.det(x), lambda x: np.linalg.eigvals(x)])
def test_array_notimpl_function_dask(func):
    x = np.random.default_rng().random((100, 100))
    y = da.from_array(x, chunks=(50, 50))
    with pytest.warns(FutureWarning, match='The `.*` function is not implemented by Dask'):
        func(y)