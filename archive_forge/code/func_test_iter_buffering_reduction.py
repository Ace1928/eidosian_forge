import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering_reduction():
    a = np.arange(6)
    b = np.array(0.0, dtype='f8').byteswap().newbyteorder()
    i = nditer([a, b], ['reduce_ok', 'buffered'], [['readonly'], ['readwrite', 'nbo']], op_axes=[[0], [-1]])
    with i:
        assert_equal(i[1].dtype, np.dtype('f8'))
        assert_(i[1].dtype != b.dtype)
        for x, y in i:
            y[...] += x
    assert_equal(b, np.sum(a))
    a = np.arange(6).reshape(2, 3)
    b = np.array([0, 0], dtype='f8').byteswap().newbyteorder()
    i = nditer([a, b], ['reduce_ok', 'external_loop', 'buffered'], [['readonly'], ['readwrite', 'nbo']], op_axes=[[0, 1], [0, -1]])
    with i:
        assert_equal(i[1].shape, (3,))
        assert_equal(i[1].strides, (0,))
        for x, y in i:
            for j in range(len(y)):
                y[j] += x[j]
    assert_equal(b, np.sum(a, axis=1))
    p = np.arange(2) + 1
    it = np.nditer([p, None], ['delay_bufalloc', 'reduce_ok', 'buffered', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[-1, 0], [-1, -1]], itershape=(2, 2))
    with it:
        it.operands[1].fill(0)
        it.reset()
        assert_equal(it[0], [1, 2, 1, 2])
    x = np.ones((7, 13, 8), np.int8)[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
    x[...] = np.arange(x.size).reshape(x.shape)
    y_base = np.arange(4 * 4, dtype=np.int8).reshape(4, 4)
    y_base_copy = y_base.copy()
    y = y_base[::2, :, None]
    it = np.nditer([y, x], ['buffered', 'external_loop', 'reduce_ok'], [['readwrite'], ['readonly']])
    with it:
        for a, b in it:
            a.fill(2)
    assert_equal(y_base[1::2], y_base_copy[1::2])
    assert_equal(y_base[::2], 2)