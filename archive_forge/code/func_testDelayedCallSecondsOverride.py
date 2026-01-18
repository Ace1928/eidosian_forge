import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def testDelayedCallSecondsOverride(self):
    """
        Test that the C{seconds} argument to DelayedCall gets used instead of
        the default timing function, if it is not None.
        """

    def seconds():
        return 10
    dc = base.DelayedCall(5, lambda: None, (), {}, lambda dc: None, lambda dc: None, seconds)
    self.assertEqual(dc.getTime(), 5)
    dc.reset(3)
    self.assertEqual(dc.getTime(), 13)