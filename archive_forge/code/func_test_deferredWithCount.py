from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_deferredWithCount(self):
    """
        In the case that the function passed to L{LoopingCall.withCount}
        returns a deferred, which does not fire before the next interval
        elapses, the function should not be run again. And if a function call
        is skipped in this fashion, the appropriate count should be
        provided.
        """
    testClock = task.Clock()
    d = defer.Deferred()
    deferredCounts = []

    def countTracker(possibleCount):
        deferredCounts.append(possibleCount)
        if len(deferredCounts) == 1:
            return d
        else:
            return None
    lc = task.LoopingCall.withCount(countTracker)
    lc.clock = testClock
    d = lc.start(0.2, now=False)
    self.assertEqual(deferredCounts, [])
    testClock.pump([0.2, 0.4])
    self.assertEqual(len(deferredCounts), 1)
    d.callback(None)
    testClock.pump([0.2])
    self.assertEqual(len(deferredCounts), 2)
    self.assertEqual(deferredCounts, [1, 3])