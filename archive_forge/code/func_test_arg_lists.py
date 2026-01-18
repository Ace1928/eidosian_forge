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
def test_arg_lists(self):
    mocks = [Mock(), MagicMock(), NonCallableMock(), NonCallableMagicMock()]

    def assert_attrs(mock):
        names = ('call_args_list', 'method_calls', 'mock_calls')
        for name in names:
            attr = getattr(mock, name)
            self.assertIsInstance(attr, _CallList)
            self.assertIsInstance(attr, list)
            self.assertEqual(attr, [])
    for mock in mocks:
        assert_attrs(mock)
        if callable(mock):
            mock()
            mock(1, 2)
            mock(a=3)
            mock.reset_mock()
            assert_attrs(mock)
        mock.foo()
        mock.foo.bar(1, a=3)
        mock.foo(1).bar().baz(3)
        mock.reset_mock()
        assert_attrs(mock)