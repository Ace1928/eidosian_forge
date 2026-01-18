import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testYieldNonDeferred(self):
    """
        Ensure that yielding a non-deferred passes it back as the
        result of the yield expression.

        @return: A L{twisted.internet.defer.Deferred}
        @rtype: L{twisted.internet.defer.Deferred}
        """

    def _test():
        yield 5
        returnValue(5)
    _test = inlineCallbacks(_test)
    return _test().addCallback(self.assertEqual, 5)