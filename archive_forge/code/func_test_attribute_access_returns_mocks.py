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
def test_attribute_access_returns_mocks(self):
    mock = Mock()
    something = mock.something
    self.assertTrue(is_instance(something, Mock), "attribute isn't a mock")
    self.assertEqual(mock.something, something, 'different attributes returned for same name')
    mock = Mock()
    mock.something.return_value = 3
    self.assertEqual(mock.something(), 3, 'method returned wrong value')
    self.assertTrue(mock.something.called, "method didn't record being called")