import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_broadcasting_errors():
    assert_raises(ValueError, nditer, [arange(2), arange(3)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(2)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(9).reshape(3, 3)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(6).reshape(2, 3), arange(4).reshape(2, 2)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(36).reshape(3, 3, 4), arange(24).reshape(2, 3, 4)], [], [['readonly']] * 2)
    assert_raises(ValueError, nditer, [arange(8).reshape(2, 4, 1), arange(24).reshape(2, 3, 4)], [], [['readonly']] * 2)
    try:
        nditer([arange(2).reshape(1, 2, 1), arange(3).reshape(1, 3), arange(6).reshape(2, 3)], [], [['readonly'], ['readonly'], ['writeonly', 'no_broadcast']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,3)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,3)' % msg)
        assert_(msg.find('(1,2,3)') >= 0, 'Message "%s" doesn\'t contain broadcast shape (1,2,3)' % msg)
    try:
        nditer([arange(6).reshape(2, 3), arange(2)], [], [['readonly'], ['readonly']], op_axes=[[0, 1], [0, np.newaxis]], itershape=(4, 3))
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,3)->(2,3)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,3)->(2,3)' % msg)
        assert_(msg.find('(2,)->(2,newaxis)') >= 0, ('Message "%s" doesn\'t contain remapped operand shape' + '(2,)->(2,newaxis)') % msg)
        assert_(msg.find('(4,3)') >= 0, 'Message "%s" doesn\'t contain itershape parameter (4,3)' % msg)
    try:
        nditer([np.zeros((2, 1, 1)), np.zeros((2,))], [], [['writeonly', 'no_broadcast'], ['readonly']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        assert_(msg.find('(2,1,1)') >= 0, 'Message "%s" doesn\'t contain operand shape (2,1,1)' % msg)
        assert_(msg.find('(2,1,2)') >= 0, 'Message "%s" doesn\'t contain the broadcast shape (2,1,2)' % msg)