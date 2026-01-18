import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_descriptors(self):

    class Foo(object):

        @classmethod
        def f(cls, a, b):
            pass

        @staticmethod
        def g(a, b):
            pass

    class Bar(Foo):
        pass

    class Baz(SomeClass, Bar):
        pass
    for spec in (Foo, Foo(), Bar, Bar(), Baz, Baz()):
        mock = create_autospec(spec)
        mock.f(1, 2)
        mock.f.assert_called_once_with(1, 2)
        mock.g(3, 4)
        mock.g.assert_called_once_with(3, 4)