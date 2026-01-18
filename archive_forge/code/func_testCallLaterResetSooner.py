from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testCallLaterResetSooner(self):
    """
        Test that calls can have their time reset to an earlier time.
        """
    events = []
    c = task.Clock()
    call = c.callLater(4, lambda a, b: events.append((a, b)), 1, b=2)
    call.reset(3)
    self.assertEqual(call.getTime(), 3)
    c.advance(3)
    self.assertEqual(events, [(1, 2)])