import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_no_inner_full_coalesce():
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        size = np.prod(shape)
        a = arange(size)
        for dirs in range(2 ** len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if 2 ** bit & dirs:
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)
            aview = a.reshape(shape)[dirs_index]
            i = nditer(aview, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            i = nditer(aview.T, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), ['external_loop'], [['readonly']])
                assert_equal(i.ndim, 1)
                assert_equal(i[0].shape, (size,))