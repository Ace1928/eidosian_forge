import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_writemasked_decref():
    arr = np.arange(10000).astype('>i,O')
    original = arr.copy()
    mask = np.random.randint(0, 2, size=10000).astype(bool)
    it = np.nditer([arr, mask], ['buffered', 'refs_ok'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['<i,O', '?'])
    singleton = object()
    if HAS_REFCOUNT:
        count = sys.getrefcount(singleton)
    for buf, mask_buf in it:
        buf[...] = (3, singleton)
    del buf, mask_buf, it
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) - count == np.count_nonzero(mask)
    assert_array_equal(arr[~mask], original[~mask])
    assert (arr[mask] == np.array((3, singleton), arr.dtype)).all()
    del arr
    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) == count