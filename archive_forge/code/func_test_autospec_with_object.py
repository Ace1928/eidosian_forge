import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_with_object(self):

    class Bar(Foo):
        extra = []
    patcher = patch(foo_name, autospec=Bar)
    mock = patcher.start()
    try:
        self.assertIsInstance(mock, Bar)
        self.assertIsInstance(mock.extra, list)
    finally:
        patcher.stop()