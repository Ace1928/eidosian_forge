import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_mocking_unbound_methods(self):

    class Foo(object):

        def foo(self, foo):
            pass
    p = patch.object(Foo, 'foo')
    mock_foo = p.start()
    Foo().foo(1)
    mock_foo.assert_called_with(1)