import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_nbo_align_contig():
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    i = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with i:
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)
    del i
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    with nditer(au, [], [['readwrite', 'updateifcopy', 'nbo']], casting='equiv') as i:
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 12345
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)
    a = np.zeros((6 * 4 + 1,), dtype='i1')[1:]
    a.dtype = 'f4'
    a[:] = np.arange(6, dtype='f4')
    assert_(not a.flags.aligned)
    i = nditer(a, [], [['readonly']])
    assert_(not i.operands[0].flags.aligned)
    assert_equal(i.operands[0], a)
    with nditer(a, [], [['readwrite', 'updateifcopy', 'aligned']]) as i:
        assert_(i.operands[0].flags.aligned)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 3
    assert_equal(a, [3] * 6)
    a = arange(12)
    i = nditer(a[:6], [], [['readonly']])
    assert_(i.operands[0].flags.contiguous)
    assert_equal(i.operands[0], a[:6])
    i = nditer(a[::2], ['buffered', 'external_loop'], [['readonly', 'contig']], buffersize=10)
    assert_(i[0].flags.contiguous)
    assert_equal(i[0], a[::2])