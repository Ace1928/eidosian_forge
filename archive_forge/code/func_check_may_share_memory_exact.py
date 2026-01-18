import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def check_may_share_memory_exact(a, b):
    got = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)
    assert_equal(np.may_share_memory(a, b), np.may_share_memory(a, b, max_work=MAY_SHARE_BOUNDS))
    a.fill(0)
    b.fill(0)
    a.fill(1)
    exact = b.any()
    err_msg = ''
    if got != exact:
        err_msg = '    ' + '\n    '.join(['base_a - base_b = %r' % (a.__array_interface__['data'][0] - b.__array_interface__['data'][0],), 'shape_a = %r' % (a.shape,), 'shape_b = %r' % (b.shape,), 'strides_a = %r' % (a.strides,), 'strides_b = %r' % (b.strides,), 'size_a = %r' % (a.size,), 'size_b = %r' % (b.size,)])
    assert_equal(got, exact, err_msg=err_msg)