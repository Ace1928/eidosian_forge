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
def test_mock_calls(self):
    mock = MagicMock()
    self.assertIs(mock.mock_calls == [], True)
    mock = MagicMock()
    mock()
    expected = [('', (), {})]
    self.assertEqual(mock.mock_calls, expected)
    mock.foo()
    expected.append(call.foo())
    self.assertEqual(mock.mock_calls, expected)
    self.assertEqual(mock.foo.mock_calls, [('', (), {})])
    mock = MagicMock()
    mock().foo(1, 2, 3, a=4, b=5)
    expected = [('', (), {}), ('().foo', (1, 2, 3), dict(a=4, b=5))]
    self.assertEqual(mock.mock_calls, expected)
    self.assertEqual(mock.return_value.foo.mock_calls, [('', (1, 2, 3), dict(a=4, b=5))])
    self.assertEqual(mock.return_value.mock_calls, [('foo', (1, 2, 3), dict(a=4, b=5))])
    mock = MagicMock()
    mock().foo.bar().baz()
    expected = [('', (), {}), ('().foo.bar', (), {}), ('().foo.bar().baz', (), {})]
    self.assertEqual(mock.mock_calls, expected)
    self.assertEqual(mock().mock_calls, call.foo.bar().baz().call_list())
    for kwargs in (dict(), dict(name='bar')):
        mock = MagicMock(**kwargs)
        int(mock.foo)
        expected = [('foo.__int__', (), {})]
        self.assertEqual(mock.mock_calls, expected)
        mock = MagicMock(**kwargs)
        mock.a()()
        expected = [('a', (), {}), ('a()', (), {})]
        self.assertEqual(mock.mock_calls, expected)
        self.assertEqual(mock.a().mock_calls, [call()])
        mock = MagicMock(**kwargs)
        mock(1)(2)(3)
        self.assertEqual(mock.mock_calls, call(1)(2)(3).call_list())
        self.assertEqual(mock().mock_calls, call(2)(3).call_list())
        self.assertEqual(mock()().mock_calls, call(3).call_list())
        mock = MagicMock(**kwargs)
        mock(1)(2)(3).a.b.c(4)
        self.assertEqual(mock.mock_calls, call(1)(2)(3).a.b.c(4).call_list())
        self.assertEqual(mock().mock_calls, call(2)(3).a.b.c(4).call_list())
        self.assertEqual(mock()().mock_calls, call(3).a.b.c(4).call_list())
        mock = MagicMock(**kwargs)
        int(mock().foo.bar().baz())
        last_call = ('().foo.bar().baz().__int__', (), {})
        self.assertEqual(mock.mock_calls[-1], last_call)
        self.assertEqual(mock().mock_calls, call.foo.bar().baz().__int__().call_list())
        self.assertEqual(mock().foo.bar().mock_calls, call.baz().__int__().call_list())
        self.assertEqual(mock().foo.bar().baz.mock_calls, call().__int__().call_list())