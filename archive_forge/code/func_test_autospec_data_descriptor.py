import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_autospec_data_descriptor(self):

    class Descriptor(object):

        def __init__(self, value):
            self.value = value

        def __get__(self, obj, cls=None):
            return self

        def __set__(self, obj, value):
            pass

    class MyProperty(property):
        pass

    class Foo(object):
        __slots__ = ['slot']

        @property
        def prop(self):
            pass

        @MyProperty
        def subprop(self):
            pass
        desc = Descriptor(42)
    foo = create_autospec(Foo)

    def check_data_descriptor(mock_attr):
        self.assertIsInstance(mock_attr, MagicMock)
        mock_attr(1, 2, 3)
        mock_attr.abc(4, 5, 6)
        mock_attr.assert_called_once_with(1, 2, 3)
        mock_attr.abc.assert_called_once_with(4, 5, 6)
    check_data_descriptor(foo.prop)
    check_data_descriptor(foo.subprop)
    check_data_descriptor(foo.slot)
    check_data_descriptor(foo.desc)