import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def testTimerLoops(self):
    l = []

    def trigger(data, number, d):
        l.append(data)
        if len(l) == number:
            d.callback(l)
    d = defer.Deferred()
    self.t = internet.TimerService(0.01, trigger, 'hello', 10, d)
    self.t.startService()
    d.addCallback(self.assertEqual, ['hello'] * 10)
    d.addCallback(lambda x: self.t.stopService())
    return d