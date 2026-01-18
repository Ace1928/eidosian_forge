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
def testGetDelayedCalls(self):
    if not hasattr(reactor, 'getDelayedCalls'):
        return
    self.checkTimers()
    self.addTimer(35, self.done)
    self.addTimer(20, self.callback)
    self.addTimer(30, self.callback)
    which = self.counter
    self.addTimer(29, self.callback)
    self.addTimer(25, self.addCallback)
    self.addTimer(26, self.callback)
    self.timers[which].cancel()
    del self.timers[which]
    self.checkTimers()
    self.deferred.addCallback(lambda x: self.checkTimers())
    return self.deferred