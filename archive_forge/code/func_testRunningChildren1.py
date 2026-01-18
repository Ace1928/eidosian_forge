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
def testRunningChildren1(self):
    s = service.Service()
    p = service.MultiService()
    s.setServiceParent(p)
    self.assertFalse(s.running)
    self.assertFalse(p.running)
    p.startService()
    self.assertTrue(s.running)
    self.assertTrue(p.running)
    p.stopService()
    self.assertFalse(s.running)
    self.assertFalse(p.running)