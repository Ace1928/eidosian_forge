import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_callDeprecationWithWrongMessage(self):
    """
        If the message passed to L{callDeprecated} doesn't match,
        L{callDeprecated} raises a test failure.
        """
    exception = self.assertRaises(self.failureException, self.callDeprecated, (self.version, 'something.wrong'), oldMethodReplaced, 1)
    self.assertIn(getVersionString(self.version), str(exception))
    self.assertIn('please use newMethod instead', str(exception))