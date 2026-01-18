import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_get_set_delete_proxy(self):

    class Something(object):
        foo = 'foo'

    class SomethingElse:
        foo = 'foo'
    for thing in (Something, SomethingElse, Something(), SomethingElse):
        proxy = _get_proxy(Something, get_only=False)

        @patch.object(proxy, 'foo', 'bar')
        def test():
            self.assertEqual(proxy.foo, 'bar')
        test()
        self.assertEqual(proxy.foo, 'foo')
        self.assertEqual(thing.foo, 'foo')
        self.assertNotIn('foo', proxy.__dict__)