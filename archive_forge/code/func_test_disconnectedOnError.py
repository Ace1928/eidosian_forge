from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorWin32Events
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID, isInIOThread
def test_disconnectedOnError(self):
    """
        If the event handler raises an exception, the event is removed from the
        reactor and the handler's C{connectionLost} method is called in the I/O
        thread and the exception is logged.
        """
    reactorThreadID = getThreadID()
    reactor = self.buildReactor()
    event = win32event.CreateEvent(None, False, False, None)
    result = []
    finished = Deferred()
    finished.addBoth(result.append)
    finished.addBoth(lambda ignored: reactor.stop())
    listener = Listener(finished)
    reactor.addEvent(event, listener, 'brokenOccurred')
    reactor.callWhenRunning(win32event.SetEvent, event)
    self.runReactor(reactor)
    self.assertIsInstance(result[0], Failure)
    result[0].trap(RuntimeError)
    self.assertEqual(reactorThreadID, listener.connLostThreadID)
    self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))