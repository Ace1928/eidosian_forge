import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertEqualNoneMessage(self):
    """
        If a message is specified as L{None}, it is not included in the error
        message of L{assertEqual}.
        """
    exceptionForNone = self.assertRaises(self.failureException, self.assertEqual, 'foo', 'bar', None)
    exceptionWithout = self.assertRaises(self.failureException, self.assertEqual, 'foo', 'bar')
    self.assertEqual(str(exceptionWithout), str(exceptionForNone))