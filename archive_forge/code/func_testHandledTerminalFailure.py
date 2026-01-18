import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testHandledTerminalFailure(self):
    """
        Create a Deferred Generator which yields a Deferred which fails and
        handles the exception which results.  Assert that the Deferred
        Generator does not errback its Deferred.
        """
    return self._genHandledTerminalFailure().addCallback(self.assertEqual, None)