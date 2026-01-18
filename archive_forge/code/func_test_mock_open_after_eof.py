import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_open_after_eof(self):
    _open = mock.mock_open(read_data='foo')
    h = _open('bar')
    h.read()
    self.assertEqual('', h.read())
    self.assertEqual('', h.read())
    self.assertEqual('', h.readline())
    self.assertEqual('', h.readline())
    self.assertEqual([], h.readlines())
    self.assertEqual([], h.readlines())