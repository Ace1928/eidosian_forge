from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_deleting_magic_methods(self):
    mock = Mock()
    self.assertFalse(hasattr(mock, '__getitem__'))
    mock.__getitem__ = Mock()
    self.assertTrue(hasattr(mock, '__getitem__'))
    del mock.__getitem__
    self.assertFalse(hasattr(mock, '__getitem__'))