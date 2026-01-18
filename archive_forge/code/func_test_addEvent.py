from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorWin32Events
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID, isInIOThread
def test_addEvent(self):
    """
        When an event which has been added to the reactor is set, the action
        associated with the event is invoked in the reactor thread.
        """
    reactorThreadID = getThreadID()
    reactor = self.buildReactor()
    event = win32event.CreateEvent(None, False, False, None)
    finished = Deferred()
    finished.addCallback(lambda ignored: reactor.stop())
    listener = Listener(finished)
    reactor.addEvent(event, listener, 'occurred')
    reactor.callWhenRunning(win32event.SetEvent, event)
    self.runReactor(reactor)
    self.assertTrue(listener.success)
    self.assertEqual(reactorThreadID, listener.logThreadID)
    self.assertEqual(reactorThreadID, listener.eventThreadID)