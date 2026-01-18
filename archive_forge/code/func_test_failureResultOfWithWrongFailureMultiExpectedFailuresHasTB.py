import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithWrongFailureMultiExpectedFailuresHasTB(self):
    """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the original
        failure traceback in the error message.
        """
    try:
        self.failureResultOf(fail(self.failure), KeyError, IOError)
    except self.failureException as e:
        self.assertIn(self.failure.getTraceback(), str(e))