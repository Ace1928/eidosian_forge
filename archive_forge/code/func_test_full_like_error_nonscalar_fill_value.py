from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_full_like_error_nonscalar_fill_value():
    x = np.full((3, 3), 1, dtype='i8')
    with pytest.raises(ValueError, match='fill_value must be scalar'):
        da.full_like(x, [100, 100], chunks=(2, 2), dtype='i8')