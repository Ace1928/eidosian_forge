from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testAdvance(self):
    """
        Test that advancing the clock will fire some calls.
        """
    events = []
    c = task.Clock()
    call = c.callLater(2, lambda: events.append(None))
    c.advance(1)
    self.assertEqual(events, [])
    c.advance(1)
    self.assertEqual(events, [None])
    self.assertFalse(call.active())