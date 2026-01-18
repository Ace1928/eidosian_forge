import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testStackUsage2(self):
    """
        Make sure we don't blow the stack when yielding immediately
        available values.
        """
    return self._genStackUsage2().addCallback(self.assertEqual, 0)