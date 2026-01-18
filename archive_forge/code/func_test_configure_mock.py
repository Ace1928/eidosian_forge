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
def test_configure_mock(self):
    mock = Mock(foo='bar')
    self.assertEqual(mock.foo, 'bar')
    mock = MagicMock(foo='bar')
    self.assertEqual(mock.foo, 'bar')
    kwargs = {'side_effect': KeyError, 'foo.bar.return_value': 33, 'foo': MagicMock()}
    mock = Mock(**kwargs)
    self.assertRaises(KeyError, mock)
    self.assertEqual(mock.foo.bar(), 33)
    self.assertIsInstance(mock.foo, MagicMock)
    mock = Mock()
    mock.configure_mock(**kwargs)
    self.assertRaises(KeyError, mock)
    self.assertEqual(mock.foo.bar(), 33)
    self.assertIsInstance(mock.foo, MagicMock)