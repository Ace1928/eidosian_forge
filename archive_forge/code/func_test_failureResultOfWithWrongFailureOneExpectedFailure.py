import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithWrongFailureOneExpectedFailure(self):
    """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        that fails with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the expected
        exception type.
        """
    try:
        self.failureResultOf(fail(self.failure), KeyError)
    except self.failureException as e:
        self.assertIn('Failure of type ({}.{}) expected on'.format(KeyError.__module__, KeyError.__name__), str(e))