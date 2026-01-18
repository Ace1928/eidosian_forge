import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_method_calls_compare_easily(self):
    mock = Mock()
    mock.something()
    self.assertEqual(mock.method_calls, [('something',)])
    self.assertEqual(mock.method_calls, [('something', (), {})])
    mock = Mock()
    mock.something('different')
    self.assertEqual(mock.method_calls, [('something', ('different',))])
    self.assertEqual(mock.method_calls, [('something', ('different',), {})])
    mock = Mock()
    mock.something(x=1)
    self.assertEqual(mock.method_calls, [('something', {'x': 1})])
    self.assertEqual(mock.method_calls, [('something', (), {'x': 1})])
    mock = Mock()
    mock.something('different', some='more')
    self.assertEqual(mock.method_calls, [('something', ('different',), {'some': 'more'})])