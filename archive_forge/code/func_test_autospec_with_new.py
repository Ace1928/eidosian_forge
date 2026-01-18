import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_with_new(self):
    patcher = patch('%s.function' % __name__, new=3, autospec=True)
    self.assertRaises(TypeError, patcher.start)
    module = sys.modules[__name__]
    patcher = patch.object(module, 'function', new=3, autospec=True)
    self.assertRaises(TypeError, patcher.start)