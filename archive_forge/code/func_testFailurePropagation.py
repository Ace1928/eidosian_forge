from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testFailurePropagation(self):
    timings = [0.3]
    clock = task.Clock()

    def foo():
        d = defer.Deferred()
        clock.callLater(0.3, d.errback, TestException())
        return d
    lc = TestableLoopingCall(clock, foo)
    d = lc.start(1)
    self.assertFailure(d, TestException)
    clock.pump(timings)
    self.assertFalse(clock.calls)
    return d