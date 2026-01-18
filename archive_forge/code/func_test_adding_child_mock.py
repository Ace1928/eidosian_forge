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
def test_adding_child_mock(self):
    for Klass in (NonCallableMock, Mock, MagicMock, NonCallableMagicMock):
        mock = Klass()
        mock.foo = Mock()
        mock.foo()
        self.assertEqual(mock.method_calls, [call.foo()])
        self.assertEqual(mock.mock_calls, [call.foo()])
        mock = Klass()
        mock.bar = Mock(name='name')
        mock.bar()
        self.assertEqual(mock.method_calls, [])
        self.assertEqual(mock.mock_calls, [])
        mock = Klass()
        mock.baz = MagicMock()()
        mock.baz()
        self.assertEqual(mock.method_calls, [])
        self.assertEqual(mock.mock_calls, [])