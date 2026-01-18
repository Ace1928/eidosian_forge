from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_methods_isolated_between_mocks(self):
    mock1 = Mock()
    mock2 = Mock()
    mock1.__iter__ = Mock(return_value=iter([]))
    self.assertEqual(list(mock1), [])
    self.assertRaises(TypeError, lambda: list(mock2))