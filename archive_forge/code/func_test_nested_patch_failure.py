import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_nested_patch_failure(self):
    original_f = Foo.f
    original_g = Foo.g

    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'missing', 1)
    @patch.object(Foo, 'f', 1)
    def thing1():
        pass

    @patch.object(Foo, 'missing', 1)
    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'f', 1)
    def thing2():
        pass

    @patch.object(Foo, 'g', 1)
    @patch.object(Foo, 'f', 1)
    @patch.object(Foo, 'missing', 1)
    def thing3():
        pass
    for func in (thing1, thing2, thing3):
        self.assertRaises(AttributeError, func)
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)