from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_methods_and_spec_set(self):

    class Iterable(object):

        def __iter__(self):
            pass
    mock = Mock(spec_set=Iterable)
    self.assertRaises(AttributeError, lambda: mock.__iter__)
    mock.__iter__ = Mock(return_value=iter([]))
    self.assertEqual(list(mock), [])

    class NonIterable(object):
        pass
    mock = Mock(spec_set=NonIterable)
    self.assertRaises(AttributeError, lambda: mock.__iter__)

    def set_int():
        mock.__int__ = Mock(return_value=iter([]))
    self.assertRaises(AttributeError, set_int)
    mock = MagicMock(spec_set=Iterable)
    self.assertEqual(list(mock), [])
    self.assertRaises(AttributeError, set_int)