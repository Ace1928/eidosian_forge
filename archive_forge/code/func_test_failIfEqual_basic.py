import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failIfEqual_basic(self):
    x, y, z = ([1], [2], [1])
    ret = self.failIfEqual(x, y)
    self.assertEqual(ret, x, 'failIfEqual should return first parameter')
    self.failUnlessRaises(self.failureException, self.failIfEqual, x, x)
    self.failUnlessRaises(self.failureException, self.failIfEqual, x, z)