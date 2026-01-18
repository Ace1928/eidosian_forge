import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_errors(self):
    for do_opt in [True, False]:
        assert_raises(ValueError, np.einsum, optimize=do_opt)
        assert_raises(ValueError, np.einsum, '', optimize=do_opt)
        assert_raises(TypeError, np.einsum, 0, 0, optimize=do_opt)
        assert_raises(TypeError, np.einsum, '', 0, out='test', optimize=do_opt)
        assert_raises(ValueError, np.einsum, '', 0, order='W', optimize=do_opt)
        assert_raises(ValueError, np.einsum, '', 0, casting='blah', optimize=do_opt)
        assert_raises(TypeError, np.einsum, '', 0, dtype='bad_data_type', optimize=do_opt)
        assert_raises(TypeError, np.einsum, '', 0, bad_arg=0, optimize=do_opt)
        assert_raises(TypeError, np.einsum, *(None,) * 63, optimize=do_opt)
        assert_raises(ValueError, np.einsum, '', 0, 0, optimize=do_opt)
        assert_raises(ValueError, np.einsum, ',', 0, [0], [0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, ',', [0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i', 0, optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'ij', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, '...i', 0, optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i...j', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i...', 0, optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'ij...', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i..', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, '.i...', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'j->..j', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'j->.j...', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i%...', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, '...j$', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i->&', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i->ij', [0, 0], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'ij->jij', [[0, 0], [0, 0]], optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'ii', np.arange(6).reshape(2, 3), optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'ii->i', np.arange(6).reshape(2, 3), optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i', np.arange(6).reshape(2, 3), optimize=do_opt)
        assert_raises(ValueError, np.einsum, 'i->i', [[0, 1], [0, 1]], out=np.arange(4).reshape(2, 2), optimize=do_opt)
        with assert_raises_regex(ValueError, "'b'"):
            a = np.ones((3, 3, 4, 5, 6))
            b = np.ones((3, 4, 5))
            np.einsum('aabcb,abc', a, b)
        assert_raises(ValueError, np.einsum, 'i->i', np.arange(6).reshape(-1, 1), optimize=do_opt, order='d')