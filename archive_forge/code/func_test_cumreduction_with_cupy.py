from __future__ import annotations
import warnings
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [np.cumsum, np.cumprod])
def test_cumreduction_with_cupy(func):
    a = cupy.ones((10, 10))
    b = da.from_array(a, chunks=(4, 4))
    result = func(b, axis=0)
    assert_eq(result, func(a, axis=0))