from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq, meta_from_array
from dask.local import get_sync
@pytest.mark.parametrize('a,b', [(da.array([1]), 1.0), (da.array([1, 2]), [1.0, 2]), (da.array([1, 2]), np.array([1.0, 2]))])
def test_assert_eq_checks_dtype(a, b):
    with pytest.raises(AssertionError, match='a and b have different dtypes'):
        assert_eq(a, b)