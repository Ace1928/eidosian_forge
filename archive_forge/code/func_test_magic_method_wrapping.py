from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_method_wrapping(self):
    mock = Mock()

    def f(self, name):
        return (self, 'fish')
    mock.__getitem__ = f
    self.assertIsNot(mock.__getitem__, f)
    self.assertEqual(mock['foo'], (mock, 'fish'))
    self.assertEqual(mock.__getitem__('foo'), (mock, 'fish'))
    mock.__getitem__ = mock
    self.assertIs(mock.__getitem__, mock)