import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('ary, prepend, append, expected', [(np.array([1, 2, 3], dtype=np.int64), None, np.nan, 'to_end'), (np.array([1, 2, 3], dtype=np.int64), np.array([5, 7, 2], dtype=np.float32), None, 'to_begin'), (np.array([1.0, 3.0, 9.0], dtype=np.int8), np.nan, np.nan, 'to_begin')])
def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
    msg = 'dtype of `{}` must be compatible'.format(expected)
    with assert_raises_regex(TypeError, msg):
        ediff1d(ary=ary, to_end=append, to_begin=prepend)