from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [da.array, da.asarray, da.asanyarray, da.tri])
def test_like_raises(func):
    assert_eq(func(1, like=func(1)), func(1))