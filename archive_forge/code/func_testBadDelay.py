from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testBadDelay(self):
    lc = task.LoopingCall(lambda: None)
    self.assertRaises(ValueError, lc.start, -1)