import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_forced_sig(self):
    a = 0.5 * np.arange(3, dtype='f8')
    assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
    with pytest.warns(DeprecationWarning):
        assert_equal(np.add(a, 0.5, sig='i', casting='unsafe'), [0, 0, 1])
    assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
    with pytest.warns(DeprecationWarning):
        assert_equal(np.add(a, 0.5, sig=('i4',), casting='unsafe'), [0, 0, 1])
    assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'), casting='unsafe'), [0, 0, 1])
    b = np.zeros((3,), dtype='f8')
    np.add(a, 0.5, out=b)
    assert_equal(b, [0.5, 1, 1.5])
    b[:] = 0
    with pytest.warns(DeprecationWarning):
        np.add(a, 0.5, sig='i', out=b, casting='unsafe')
    assert_equal(b, [0, 0, 1])
    b[:] = 0
    np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
    assert_equal(b, [0, 0, 1])
    b[:] = 0
    with pytest.warns(DeprecationWarning):
        np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
    assert_equal(b, [0, 0, 1])
    b[:] = 0
    np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
    assert_equal(b, [0, 0, 1])