from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_new_greenlet(self):
    self.assertEqual(-15, _test_extension.test_new_greenlet(lambda: -15))