import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_inherits(self):
    FooClass = Foo
    patcher = patch(foo_name, autospec=True)
    mock = patcher.start()
    try:
        self.assertIsInstance(mock, FooClass)
        self.assertIsInstance(mock(3), FooClass)
    finally:
        patcher.stop()