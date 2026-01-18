import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_name_preserved(self):
    foo = {}

    @patch('%s.SomeClass' % __name__, object())
    @patch('%s.SomeClass' % __name__, object(), autospec=True)
    @patch.object(SomeClass, object())
    @patch.dict(foo)
    def some_name():
        pass
    self.assertEqual(some_name.__name__, 'some_name')