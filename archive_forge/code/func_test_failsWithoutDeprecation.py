import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failsWithoutDeprecation(self):
    """
        L{callDeprecated} raises a test failure if the callable is not
        deprecated.
        """

    def notDeprecated():
        pass
    exception = self.assertRaises(self.failureException, self.callDeprecated, self.version, notDeprecated)
    self.assertEqual('%r is not deprecated.' % notDeprecated, str(exception))