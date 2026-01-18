import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_common_dtype():
    i = nditer([array([3], dtype='f4'), array([0], dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='same_kind')
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.dtypes[1], np.dtype('f4'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('u4'))
    assert_equal(i.dtypes[1], np.dtype('u4'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4')], ['common_dtype'], [['readonly', 'copy']] * 2, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i8'))
    assert_equal(i.dtypes[1], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), array([2j], dtype='c8'), array([9], dtype='f8')], ['common_dtype'], [['readonly', 'copy']] * 4, casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    assert_equal(i.dtypes[3], np.dtype('c16'))
    assert_equal(i.value, (3, -12, 2j, 9))
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], [], [['readonly', 'copy'], ['writeonly', 'allocate'], ['writeonly']], casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.dtypes[1], np.dtype('i4'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], ['common_dtype'], [['readonly', 'copy'], ['writeonly', 'allocate'], ['writeonly']], casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))