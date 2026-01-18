from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_len_contains_iter(self):
    mock = Mock()
    self.assertRaises(TypeError, len, mock)
    self.assertRaises(TypeError, iter, mock)
    self.assertRaises(TypeError, lambda: 'foo' in mock)
    mock.__len__ = lambda s: 6
    self.assertEqual(len(mock), 6)
    mock.__contains__ = lambda s, o: o == 3
    self.assertIn(3, mock)
    self.assertNotIn(6, mock)
    mock.__iter__ = lambda s: iter('foobarbaz')
    self.assertEqual(list(mock), list('foobarbaz'))