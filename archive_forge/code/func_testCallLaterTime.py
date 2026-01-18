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
def testCallLaterTime(self):
    d = reactor.callLater(10, lambda: None)
    try:
        self.assertTrue(d.getTime() - (time.time() + 10) < 1)
    finally:
        d.cancel()