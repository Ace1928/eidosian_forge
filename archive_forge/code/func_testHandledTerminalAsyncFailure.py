import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testHandledTerminalAsyncFailure(self):
    """
        Just like testHandledTerminalFailure, only with a Deferred which fires
        asynchronously with an error.
        """
    d = defer.Deferred()
    deferredGeneratorResultDeferred = self._genHandledTerminalAsyncFailure(d)
    d.errback(TerminalException('Handled Terminal Failure'))
    return deferredGeneratorResultDeferred.addCallback(self.assertEqual, None)