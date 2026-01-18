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
def test_mock_parents(self):
    for Klass in (Mock, MagicMock):
        m = Klass()
        original_repr = repr(m)
        m.return_value = m
        self.assertIs(m(), m)
        self.assertEqual(repr(m), original_repr)
        m.reset_mock()
        self.assertIs(m(), m)
        self.assertEqual(repr(m), original_repr)
        m = Klass()
        m.b = m.a
        self.assertIn("name='mock.a'", repr(m.b))
        self.assertIn("name='mock.a'", repr(m.a))
        m.reset_mock()
        self.assertIn("name='mock.a'", repr(m.b))
        self.assertIn("name='mock.a'", repr(m.a))
        m = Klass()
        original_repr = repr(m)
        m.a = m()
        m.a.return_value = m
        self.assertEqual(repr(m), original_repr)
        self.assertEqual(repr(m.a()), original_repr)