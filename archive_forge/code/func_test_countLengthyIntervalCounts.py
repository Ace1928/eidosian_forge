from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_countLengthyIntervalCounts(self):
    """
        L{LoopingCall.withCount} counts only calls that were expected to be
        made.  So, if more than one, but less than two intervals pass between
        invocations, it won't increase the count above 1.  For example, a
        L{LoopingCall} with interval T expects to be invoked at T, 2T, 3T, etc.
        However, the reactor takes some time to get around to calling it, so in
        practice it will be called at T+something, 2T+something, 3T+something;
        and due to other things going on in the reactor, "something" is
        variable.  It won't increase the count unless "something" is greater
        than T.  So if the L{LoopingCall} is invoked at T, 2.75T, and 3T,
        the count has not increased, even though the distance between
        invocation 1 and invocation 2 is 1.75T.
        """
    times = []
    clock = task.Clock()

    def aCallback(count):
        times.append((clock.seconds(), count))
    call = task.LoopingCall.withCount(aCallback)
    call.clock = clock
    INTERVAL = 0.5
    REALISTIC_DELAY = 0.01
    call.start(INTERVAL)
    self.assertEqual(times.pop(), (0, 1))
    clock.advance(INTERVAL + REALISTIC_DELAY)
    self.assertEqual(times.pop(), (INTERVAL + REALISTIC_DELAY, 1))
    clock.advance(INTERVAL * 1.75)
    self.assertEqual(times.pop(), (2.75 * INTERVAL + REALISTIC_DELAY, 1))
    clock.advance(INTERVAL * 0.25)
    self.assertEqual(times.pop(), (3.0 * INTERVAL + REALISTIC_DELAY, 1))