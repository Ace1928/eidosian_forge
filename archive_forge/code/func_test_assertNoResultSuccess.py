import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertNoResultSuccess(self):
    """
        When passed a coroutine which currently has a success result (see
        L{test_withSuccessResult}), L{SynchronousTestCase.assertNoResult}
        raises L{SynchronousTestCase.failureException}.
        """
    self.assertRaises(self.failureException, self.assertNoResult, self.successResult())