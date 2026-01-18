import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithWrongFailureMultiExpectedFailures(self):
    """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the expected
        exception types in the error message.
        """
    try:
        self.failureResultOf(fail(self.failure), KeyError, IOError)
    except self.failureException as e:
        self.assertIn('Failure of type ({}.{} or {}.{}) expected on'.format(KeyError.__module__, KeyError.__name__, IOError.__module__, IOError.__name__), str(e))