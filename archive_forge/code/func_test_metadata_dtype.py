import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.parametrize('dt, fail', [(np.dtype({'names': ['a', 'b'], 'formats': [float, np.dtype('S3', metadata={'some': 'stuff'})]}), True), (np.dtype(int, metadata={'some': 'stuff'}), False), (np.dtype([('subarray', (int, (2,)))], metadata={'some': 'stuff'}), False), (np.dtype({'names': ['a', 'b'], 'formats': [float, np.dtype({'names': ['c'], 'formats': [np.dtype(int, metadata={})]})]}), False)])
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
def test_metadata_dtype(dt, fail):
    arr = np.ones(10, dtype=dt)
    buf = BytesIO()
    with assert_warns(UserWarning):
        np.save(buf, arr)
    buf.seek(0)
    if fail:
        with assert_raises(ValueError):
            np.load(buf)
    else:
        arr2 = np.load(buf)
        from numpy.lib.utils import drop_metadata
        assert_array_equal(arr, arr2)
        assert drop_metadata(arr.dtype) is not arr.dtype
        assert drop_metadata(arr2.dtype) is arr2.dtype