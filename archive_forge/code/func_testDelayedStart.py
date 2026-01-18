from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testDelayedStart(self):
    timings = [0.05, 0.1, 0.1]
    clock = task.Clock()
    L = []
    lc = TestableLoopingCall(clock, L.append, None)
    d = lc.start(0.1, now=False)
    theResult = []

    def saveResult(result):
        theResult.append(result)
    d.addCallback(saveResult)
    clock.pump(timings)
    self.assertEqual(len(L), 2, 'got %d iterations, not 2' % (len(L),))
    lc.stop()
    self.assertIs(theResult[0], lc)
    self.assertFalse(clock.calls)