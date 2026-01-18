import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithExpectedException(self):
    """
        When passed a coroutine which currently has an exception result (ie, if
        converted into a L{Deferred}, L{Deferred.addErrback} would cause the
        added errback to be called before C{addErrback} returns),
        L{SynchronousTestCase.failureResultOf} returns a L{Failure} containing
        that exception, if the exception type is expected.
        """
    self.assertEqual(self.failure.value, self.failureResultOf(self.raisesException(), self.failure.type, KeyError).value)