import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_magic_methods(self):

    class BuiltinSubclass(list):
        attr = {}
    mock = create_autospec(BuiltinSubclass)
    self.assertEqual(list(mock), [])
    self.assertRaises(TypeError, int, mock)
    self.assertRaises(TypeError, int, mock.attr)
    self.assertEqual(list(mock), [])
    self.assertIsInstance(mock['foo'], MagicMock)
    self.assertIsInstance(mock.attr['foo'], MagicMock)