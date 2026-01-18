import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_name(self):
    patcher = patch(foo_name, autospec=True)
    mock = patcher.start()
    try:
        self.assertIn(" name='Foo'", repr(mock))
        self.assertIn(" name='Foo.f'", repr(mock.f))
        self.assertIn(" name='Foo()'", repr(mock(None)))
        self.assertIn(" name='Foo().f'", repr(mock(None).f))
    finally:
        patcher.stop()