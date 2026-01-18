import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertNoResultPropagatesSuccess(self):
    """
        When passed a coroutine awaiting a L{Deferred} with no current result,
        which is then fired with a success result,
        L{SynchronousTestCase.assertNoResult} doesn't modify the result of the
        L{Deferred}.
        """
    d = Deferred()

    async def noCurrentResult():
        return await d
    c = noCurrentResult()
    self.assertNoResult(d)
    d.callback(self.result)
    self.assertEqual(self.result, self.successResultOf(c))