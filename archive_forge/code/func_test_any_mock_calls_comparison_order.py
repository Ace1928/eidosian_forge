import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_any_mock_calls_comparison_order(self):
    mock = Mock()
    d = datetime.now()

    class Foo(object):

        def __eq__(self, other):
            return False

        def __ne__(self, other):
            return True
    for d in (datetime.now(), Foo()):
        mock.reset_mock()
        mock(d, foo=d, bar=d)
        mock.method(d, zinga=d, alpha=d)
        mock().method(a1=d, z99=d)
        expected = [call(ANY, foo=ANY, bar=ANY), call.method(ANY, zinga=ANY, alpha=ANY), call(), call().method(a1=ANY, z99=ANY)]
        self.assertEqual(expected, mock.mock_calls)
        self.assertEqual(mock.mock_calls, expected)