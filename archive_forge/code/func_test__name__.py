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
def test__name__(self):
    mock = Mock()
    self.assertRaises(AttributeError, lambda: mock.__name__)
    mock.__name__ = 'foo'
    self.assertEqual(mock.__name__, 'foo')