import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering():
    arrays = []
    arrays.append(np.arange(24, dtype='c16').reshape(2, 3, 4).T.newbyteorder().byteswap())
    arrays.append(np.arange(10, dtype='f4'))
    a = np.zeros((4 * 16 + 1,), dtype='i1')[1:]
    a.dtype = 'i4'
    a[:] = np.arange(16, dtype='i4')
    arrays.append(a)
    arrays.append(np.arange(120, dtype='i4').reshape(5, 3, 2, 4).T)
    for a in arrays:
        for buffersize in (1, 2, 3, 5, 8, 11, 16, 1024):
            vals = []
            i = nditer(a, ['buffered', 'external_loop'], [['readonly', 'nbo', 'aligned']], order='C', casting='equiv', buffersize=buffersize)
            while not i.finished:
                assert_(i[0].size <= buffersize)
                vals.append(i[0].copy())
                i.iternext()
            assert_equal(np.concatenate(vals), a.ravel(order='C'))