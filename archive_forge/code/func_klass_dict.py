import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@patch.dict(foo)
@classmethod
def klass_dict(cls):
    self.assertIs(cls, Something)