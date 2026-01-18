import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
def test_invalid_structured_dtypes(self):
    assert_raises(ValueError, np.dtype, ('O', [('name', 'i8')]))
    assert_raises(ValueError, np.dtype, ('i8', [('name', 'O')]))
    assert_raises(ValueError, np.dtype, ('i8', [('name', [('name', 'O')])]))
    assert_raises(ValueError, np.dtype, ([('a', 'i4'), ('b', 'i4')], 'O'))
    assert_raises(ValueError, np.dtype, ('i8', 'O'))
    assert_raises(ValueError, np.dtype, ('i', {'name': ('i', 0, 'title', 'oops')}))
    assert_raises(ValueError, np.dtype, ('i', {'name': ('i', 'wrongtype', 'title')}))
    assert_raises(ValueError, np.dtype, ([('a', 'O'), ('b', 'O')], [('c', 'O'), ('d', 'O')]))
    a = np.ones(1, dtype=('O', [('name', 'O')]))
    assert_equal(a[0], 1)
    assert a[0] is a.item()
    assert type(a[0]) is int