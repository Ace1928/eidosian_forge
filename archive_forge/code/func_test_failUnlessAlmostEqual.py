import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessAlmostEqual(self):
    precision = 5
    x = 8.000001
    y = 8.00001
    z = 8.000002
    self.failUnlessAlmostEqual(x, x, precision)
    ret = self.failUnlessAlmostEqual(x, z, precision)
    self.assertEqual(ret, x, 'failUnlessAlmostEqual should return first parameter (%r, %r)' % (ret, x))
    self.failUnlessRaises(self.failureException, self.failUnlessAlmostEqual, x, y, precision)