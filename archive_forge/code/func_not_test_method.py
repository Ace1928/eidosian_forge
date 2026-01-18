import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def not_test_method(other_self):
    self.assertEqual(PTModule.something, sentinel.Something, 'non-test method patched')