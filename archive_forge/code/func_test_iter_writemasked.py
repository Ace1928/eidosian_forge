import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.parametrize('a', [np.zeros((3,), dtype='f8'), np.zeros((9876, 3 * 5), dtype='f8')[::2, :], np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, :], np.zeros((9,), dtype='f8')[::3], np.zeros((9876, 3 * 10), dtype='f8')[::2, ::5], np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, ::-1]])
def test_iter_writemasked(a):
    shape = a.shape
    reps = shape[-1] // 3
    msk = np.empty(shape, dtype=bool)
    msk[...] = [True, True, False] * reps
    it = np.nditer([a, msk], [], [['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    with it:
        for x, m in it:
            x[...] = 1
    assert_equal(a, np.broadcast_to([1, 1, 1] * reps, shape))
    it = np.nditer([a, msk], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    is_buffered = True
    with it:
        for x, m in it:
            x[...] = 2.5
            if np.may_share_memory(x, a):
                is_buffered = False
    if not is_buffered:
        assert_equal(a, np.broadcast_to([2.5, 2.5, 2.5] * reps, shape))
    else:
        assert_equal(a, np.broadcast_to([2.5, 2.5, 1] * reps, shape))
        a[...] = 2.5
    it = np.nditer([a, msk], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['i8', None], casting='unsafe')
    with it:
        for x, m in it:
            x[...] = 3
    assert_equal(a, np.broadcast_to([3, 3, 2.5] * reps, shape))