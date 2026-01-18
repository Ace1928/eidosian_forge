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
def test_class_assignable(self):
    for mock in (Mock(), MagicMock()):
        self.assertNotIsInstance(mock, int)
        mock.__class__ = int
        self.assertIsInstance(mock, int)
        mock.foo