from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
@unittest.skipIf(six.PY3, 'no __cmp__ in Python 3')
def test_non_default_magic_methods(self):
    mock = MagicMock()
    self.assertRaises(AttributeError, lambda: mock.__cmp__)
    mock = Mock()
    mock.__cmp__ = lambda s, o: 0
    self.assertEqual(mock, object())