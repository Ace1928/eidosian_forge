from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testEveryIteration(self):
    ran = []

    def foo():
        ran.append(None)
        if len(ran) > 5:
            lc.stop()
    lc = task.LoopingCall(foo)
    d = lc.start(0)

    def stopped(ign):
        self.assertEqual(len(ran), 6)
    return d.addCallback(stopped)