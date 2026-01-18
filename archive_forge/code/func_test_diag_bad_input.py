from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('k', [0, 3, -3, 8])
def test_diag_bad_input(k):
    v = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    with pytest.raises(ValueError, match='Array must be 1d or 2d only'):
        da.diag(v, k)
    v = da.arange(2 * 3 * 4).reshape((2, 3, 4))
    with pytest.raises(ValueError, match='Array must be 1d or 2d only'):
        da.diag(v, k)
    v = 1
    with pytest.raises(TypeError, match='v must be a dask array or numpy array'):
        da.diag(v, k)