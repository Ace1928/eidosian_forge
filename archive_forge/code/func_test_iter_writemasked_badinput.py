import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_writemasked_badinput():
    a = np.zeros((2, 3))
    b = np.zeros((3,))
    m = np.array([[True, True, False], [False, True, False]])
    m2 = np.array([True, True, False])
    m3 = np.array([0, 1, 1], dtype='u1')
    mbad1 = np.array([0, 1, 1], dtype='i1')
    mbad2 = np.array([0, 1, 1], dtype='f4')
    assert_raises(ValueError, nditer, [a, m], [], [['readwrite', 'writemasked'], ['readonly']])
    assert_raises(ValueError, nditer, [a, m], [], [['readonly', 'writemasked'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, m], [], [['readonly'], ['readwrite', 'arraymask', 'writemasked']])
    assert_raises(ValueError, nditer, [a, m, m2], [], [['readwrite', 'writemasked'], ['readonly', 'arraymask'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, m], [], [['readwrite'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, b, m], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    np.nditer([a, b, m2], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readonly', 'arraymask']])
    assert_raises(ValueError, nditer, [a, b, m2], ['reduce_ok'], [['readonly'], ['readwrite', 'writemasked'], ['readwrite', 'arraymask']])
    np.nditer([a, m3], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')
    assert_raises(TypeError, np.nditer, [a, mbad1], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')
    assert_raises(TypeError, np.nditer, [a, mbad2], ['buffered'], [['readwrite', 'writemasked'], ['readonly', 'arraymask']], op_dtypes=['f4', None], casting='same_kind')