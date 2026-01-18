import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_failure(self):
    original_f = Foo.f
    original_g = Foo.g
    original_foo = Foo.foo

    def crasher():
        raise NameError('crasher')

    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'foo', new_callable=crasher)
    @patch.object(Foo, 'f', 1)
    def thing1():
        pass

    @patch.object(Foo, 'foo', new_callable=crasher)
    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'f', 1)
    def thing2():
        pass

    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'f', 1)
    @patch.object(Foo, 'foo', new_callable=crasher)
    def thing3():
        pass
    for func in (thing1, thing2, thing3):
        self.assertRaises(NameError, func)
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)
        self.assertEqual(Foo.foo, original_foo)