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
def test_assert_called_once_with(self):
    mock = Mock()
    mock()
    mock.assert_called_once_with()
    mock()
    self.assertRaises(AssertionError, mock.assert_called_once_with)
    mock.reset_mock()
    self.assertRaises(AssertionError, mock.assert_called_once_with)
    mock('foo', 'bar', baz=2)
    mock.assert_called_once_with('foo', 'bar', baz=2)
    mock.reset_mock()
    mock('foo', 'bar', baz=2)
    self.assertRaises(AssertionError, lambda: mock.assert_called_once_with('bob', 'bar', baz=2))