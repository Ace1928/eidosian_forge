import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertNoResultPropagatesLaterFailure(self):
    """
        When passed a coroutine awaiting a L{Deferred} with no current result,
        which is then fired with a L{Failure} result,
        L{SynchronousTestCase.assertNoResult} doesn't modify the result of the
        L{Deferred}.
        """
    f = Failure(self.exception)
    d = Deferred()

    async def noCurrentResult():
        return await d
    c = noCurrentResult()
    self.assertNoResult(d)
    d.errback(f)
    self.assertEqual(f.value, self.failureResultOf(c).value)