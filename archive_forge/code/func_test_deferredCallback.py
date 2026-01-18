import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_deferredCallback(self):
    """
        If the function called by L{utils.runWithWarningsSuppressed} returns a
        C{Deferred}, the warning filters aren't removed until the Deferred
        fires.
        """
    filters = [(('ignore', '.*foo.*'), {}), (('ignore', '.*bar.*'), {})]
    result = Deferred()
    self.runWithWarningsSuppressed(filters, lambda: result)
    warnings.warn('ignore foo')
    result.callback(3)
    warnings.warn('ignore foo 2')
    self.assertEqual(['ignore foo 2'], [w['message'] for w in self.flushWarnings()])