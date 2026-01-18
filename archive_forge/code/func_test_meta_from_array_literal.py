from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq, meta_from_array
from dask.local import get_sync
@pytest.mark.parametrize('meta', ['', 'str', '', 'str', b'', b'str'])
@pytest.mark.parametrize('dtype', [None, 'bool', 'int', 'float'])
def test_meta_from_array_literal(meta, dtype):
    if dtype is None:
        assert meta_from_array(meta, dtype=dtype).dtype.kind in 'SU'
    else:
        assert meta_from_array(meta, dtype=dtype).dtype == np.array([], dtype=dtype).dtype