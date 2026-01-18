from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testWaitDeferred(self):
    timings = [0.2, 0.8]
    clock = task.Clock()

    def foo():
        d = defer.Deferred()
        d.addCallback(lambda _: lc.stop())
        clock.callLater(1, d.callback, None)
        return d
    lc = TestableLoopingCall(clock, foo)
    lc.start(0.2)
    clock.pump(timings)
    self.assertFalse(clock.calls)