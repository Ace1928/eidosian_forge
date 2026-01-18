from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_callbackTimeSkips(self):
    """
        When more time than the defined interval passes during the execution
        of a callback, L{LoopingCall} should schedule the next call for the
        next interval which is still in the future.
        """
    times = []
    callDuration = None
    clock = task.Clock()

    def aCallback():
        times.append(clock.seconds())
        clock.advance(callDuration)
    call = task.LoopingCall(aCallback)
    call.clock = clock
    callDuration = 2
    call.start(0.5)
    self.assertEqual(times, [0])
    self.assertEqual(clock.seconds(), callDuration)
    clock.advance(0)
    self.assertEqual(times, [0])
    callDuration = 1
    clock.advance(0.5)
    self.assertEqual(times, [0, 2.5])
    self.assertEqual(clock.seconds(), 3.5)
    clock.advance(0)
    self.assertEqual(times, [0, 2.5])
    callDuration = 0
    clock.advance(0.5)
    self.assertEqual(times, [0, 2.5, 4])
    self.assertEqual(clock.seconds(), 4)