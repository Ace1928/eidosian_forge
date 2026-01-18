import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessIdentical(self):
    x, y, z = ([1], [1], [2])
    ret = self.failUnlessIdentical(x, x)
    self.assertEqual(ret, x, 'failUnlessIdentical should return first parameter')
    self.failUnlessRaises(self.failureException, self.failUnlessIdentical, x, y)
    self.failUnlessRaises(self.failureException, self.failUnlessIdentical, x, z)