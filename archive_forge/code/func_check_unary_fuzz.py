import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def check_unary_fuzz(self, operation, get_out_axis_size, dtype=np.int16, count=5000):
    shapes = [7, 13, 8, 21, 29, 32]
    rng = np.random.RandomState(1234)
    for ndim in range(1, 6):
        x = rng.randint(0, 2 ** 16, size=shapes[:ndim]).astype(dtype)
        it = iter_random_view_pairs(x, same_steps=False, equal_size=True)
        min_count = count // (ndim + 1) ** 2
        overlapping = 0
        while overlapping < min_count:
            a, b = next(it)
            a_orig = a.copy()
            b_orig = b.copy()
            if get_out_axis_size is None:
                assert_copy_equivalent(operation, [a], out=b)
                if np.shares_memory(a, b):
                    overlapping += 1
            else:
                for axis in itertools.chain(range(ndim), [None]):
                    a[...] = a_orig
                    b[...] = b_orig
                    outsize, scalarize = get_out_axis_size(a, b, axis)
                    if outsize == 'skip':
                        continue
                    sl = [slice(None)] * ndim
                    if axis is None:
                        if outsize is None:
                            sl = [slice(0, 1)] + [0] * (ndim - 1)
                        else:
                            sl = [slice(0, outsize)] + [0] * (ndim - 1)
                    elif outsize is None:
                        k = b.shape[axis] // 2
                        if ndim == 1:
                            sl[axis] = slice(k, k + 1)
                        else:
                            sl[axis] = k
                    else:
                        assert b.shape[axis] >= outsize
                        sl[axis] = slice(0, outsize)
                    b_out = b[tuple(sl)]
                    if scalarize:
                        b_out = b_out.reshape([])
                    if np.shares_memory(a, b_out):
                        overlapping += 1
                    assert_copy_equivalent(operation, [a], out=b_out, axis=axis)