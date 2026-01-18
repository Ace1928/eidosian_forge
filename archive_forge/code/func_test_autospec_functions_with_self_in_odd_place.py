import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_autospec_functions_with_self_in_odd_place(self):

    class Foo(object):

        def f(a, self):
            pass
    a = create_autospec(Foo)
    a.f(10)
    a.f.assert_called_with(10)
    a.f.assert_called_with(self=10)
    a.f(self=10)
    a.f.assert_called_with(10)
    a.f.assert_called_with(self=10)