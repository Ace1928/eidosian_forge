from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [np.array, np.asarray, np.asanyarray])
def test_like_with_numpy_func(func):
    assert_eq(func(1, like=da.array(1)), func(1))