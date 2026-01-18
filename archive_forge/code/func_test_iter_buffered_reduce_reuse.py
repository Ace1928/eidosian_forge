import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.slow
def test_iter_buffered_reduce_reuse():
    a = np.arange(2 * 3 ** 5)[3 ** 5:3 ** 5 + 1]
    flags = ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok', 'refs_ok']
    op_flags = [('readonly',), ('readwrite', 'allocate')]
    op_axes_list = [[(0, 1, 2), (0, 1, -1)], [(0, 1, 2), (0, -1, -1)]]
    op_dtypes = [float, a.dtype]

    def get_params():
        for xs in range(-3 ** 2, 3 ** 2 + 1):
            for ys in range(xs, 3 ** 2 + 1):
                for op_axes in op_axes_list:
                    strides = (xs * a.itemsize, ys * a.itemsize, a.itemsize)
                    arr = np.lib.stride_tricks.as_strided(a, (3, 3, 3), strides)
                    for skip in [0, 1]:
                        yield (arr, op_axes, skip)
    for arr, op_axes, skip in get_params():
        nditer2 = np.nditer([arr.copy(), None], op_axes=op_axes, flags=flags, op_flags=op_flags, op_dtypes=op_dtypes)
        with nditer2:
            nditer2.operands[-1][...] = 0
            nditer2.reset()
            nditer2.iterindex = skip
            for a2_in, b2_in in nditer2:
                b2_in += a2_in.astype(np.int_)
            comp_res = nditer2.operands[-1]
        for bufsize in range(0, 3 ** 3):
            nditer1 = np.nditer([arr, None], op_axes=op_axes, flags=flags, op_flags=op_flags, buffersize=bufsize, op_dtypes=op_dtypes)
            with nditer1:
                nditer1.operands[-1][...] = 0
                nditer1.reset()
                nditer1.iterindex = skip
                for a1_in, b1_in in nditer1:
                    b1_in += a1_in.astype(np.int_)
                res = nditer1.operands[-1]
            assert_array_equal(res, comp_res)