import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_create_autospec_none(self):

    class Foo(object):
        bar = None
    mock = create_autospec(Foo)
    none = mock.bar
    self.assertNotIsInstance(none, type(None))
    none.foo()
    none.foo.assert_called_once_with()