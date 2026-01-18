import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_copy_if_overlap():
    for flag in ['readonly', 'writeonly', 'readwrite']:
        a = arange(10)
        i = nditer([a], ['copy_if_overlap'], [[flag]])
        with i:
            assert_(i.operands[0] is a)
    x = arange(10)
    a = x[1:]
    b = x[:-1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(not np.shares_memory(*i.operands))
    x = arange(10)
    a = x
    b = x
    i = nditer([a, b], ['copy_if_overlap'], [['readonly', 'overlap_assume_elementwise'], ['readwrite', 'overlap_assume_elementwise']])
    with i:
        assert_(i.operands[0] is a and i.operands[1] is b)
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(i.operands[0] is a and (not np.shares_memory(i.operands[1], b)))
    x = arange(10)
    a = x[::2]
    b = x[1::2]
    i = nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']])
    assert_(i.operands[0] is a and i.operands[1] is b)
    x = arange(4, dtype=np.int8)
    a = x[3:]
    b = x.view(np.int32)[:1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']]) as i:
        assert_(not np.shares_memory(*i.operands))
    for flag in ['writeonly', 'readwrite']:
        x = np.ones([10, 10])
        a = x
        b = x.T
        c = x
        with nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['readonly'], [flag]]) as i:
            a2, b2, c2 = i.operands
            assert_(not np.shares_memory(a2, c2))
            assert_(not np.shares_memory(b2, c2))
    x = np.ones([10, 10])
    a = x
    b = x.T
    c = x
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['readonly'], ['readonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)
    x = np.ones([10, 10])
    a = x
    b = np.ones([10, 10])
    c = x.T
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['writeonly'], ['readonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)
    x = np.arange(7)
    a = x[:3]
    b = x[3:6]
    c = x[4:7]
    i = nditer([a, b, c], ['copy_if_overlap'], [['readonly'], ['writeonly'], ['writeonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)