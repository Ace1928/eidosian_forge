import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_has_calls_nested_spec(self):

    class Something:

        def __init__(self):
            pass

        def meth(self, a, b, c, d=None):
            pass

        class Foo:

            def __init__(self, a):
                pass

            def meth1(self, a, b):
                pass
    mock_class = create_autospec(Something)
    for m in [mock_class, mock_class()]:
        m.meth(1, 2, 3, d=1)
        m.assert_has_calls([call.meth(1, 2, 3, d=1)])
        m.assert_has_calls([call.meth(1, 2, 3, 1)])
    mock_class.reset_mock()
    for m in [mock_class, mock_class()]:
        self.assertRaises(AssertionError, m.assert_has_calls, [call.Foo()])
        m.Foo(1).meth1(1, 2)
        m.assert_has_calls([call.Foo(1), call.Foo(1).meth1(1, 2)])
        m.Foo.assert_has_calls([call(1), call().meth1(1, 2)])
    mock_class.reset_mock()
    invalid_calls = [call.meth(1), call.non_existent(1), call.Foo().non_existent(1), call.Foo().meth(1, 2, 3, 4)]
    for kall in invalid_calls:
        self.assertRaises(AssertionError, mock_class.assert_has_calls, [kall])