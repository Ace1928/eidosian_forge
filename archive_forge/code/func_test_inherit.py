import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_inherit(self):

    class Foo(object):
        a = 3
    Foo.Foo = Foo
    mock = create_autospec(Foo)
    instance = mock()
    self.assertRaises(AttributeError, getattr, instance, 'b')
    attr_instance = mock.Foo()
    self.assertRaises(AttributeError, getattr, attr_instance, 'b')
    mock = create_autospec(Foo())
    self.assertRaises(AttributeError, getattr, mock, 'b')
    self.assertRaises(TypeError, mock)
    call_result = mock.Foo()
    self.assertRaises(AttributeError, getattr, call_result, 'b')