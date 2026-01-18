import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_from_unicode_array(self):
    A = np.array([['abc', 'Sigma Σ'], ['long   ', '0123456789']])
    assert_equal(A.dtype.type, np.str_)
    B = np.char.array(A)
    assert_array_equal(B, A)
    assert_equal(B.dtype, A.dtype)
    assert_equal(B.shape, A.shape)
    B = np.char.array(A, **kw_unicode_true)
    assert_array_equal(B, A)
    assert_equal(B.dtype, A.dtype)
    assert_equal(B.shape, A.shape)

    def fail():
        np.char.array(A, **kw_unicode_false)
    assert_raises(UnicodeEncodeError, fail)