import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testReturnNoValue(self):
    """Ensure a standard python return results in a None result."""

    def _noReturn():
        yield 5
        return
    _noReturn = inlineCallbacks(_noReturn)
    return _noReturn().addCallback(self.assertEqual, None)