from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testCallLaterResetLater(self):
    """
        Test that calls can have their time reset to a later time.
        """
    events = []
    c = task.Clock()
    call = c.callLater(2, lambda a, b: events.append((a, b)), 1, b=2)
    c.advance(1)
    call.reset(3)
    self.assertEqual(call.getTime(), 4)
    c.advance(2)
    self.assertEqual(events, [])
    c.advance(1)
    self.assertEqual(events, [(1, 2)])