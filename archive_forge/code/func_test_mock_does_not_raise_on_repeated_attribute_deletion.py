import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_does_not_raise_on_repeated_attribute_deletion(self):
    for mock in (Mock(), MagicMock(), NonCallableMagicMock(), NonCallableMock()):
        mock.foo = 3
        self.assertTrue(hasattr(mock, 'foo'))
        self.assertEqual(mock.foo, 3)
        del mock.foo
        self.assertFalse(hasattr(mock, 'foo'))
        mock.foo = 4
        self.assertTrue(hasattr(mock, 'foo'))
        self.assertEqual(mock.foo, 4)
        del mock.foo
        self.assertFalse(hasattr(mock, 'foo'))