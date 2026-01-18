import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.skipif(IS_PYPY, reason='flaky on PyPy')
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='test requires 64-bit system')
@pytest.mark.slow
@requires_memory(free_bytes=2 * 2 ** 30)
def test_large_archive(tmpdir):
    shape = (2 ** 30, 2)
    try:
        a = np.empty(shape, dtype=np.uint8)
    except MemoryError:
        pytest.skip('Could not create large file')
    fname = os.path.join(tmpdir, 'large_archive')
    with open(fname, 'wb') as f:
        np.savez(f, arr=a)
    del a
    with open(fname, 'rb') as f:
        new_a = np.load(f)['arr']
    assert new_a.shape == shape