import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failIfIdenticalNegative(self):
    """
        C{failIfIdentical} raises C{failureException} if its first and second
        arguments are the same object.
        """
    x = object()
    self.failUnlessRaises(self.failureException, self.failIfIdentical, x, x)