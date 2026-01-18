from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorWin32Events
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID, isInIOThread
def test_ioThreadDoesNotChange(self):
    """
        Using L{IReactorWin32Events.addEvent} does not change which thread is
        reported as the I/O thread.
        """
    results = []

    def check(ignored):
        results.append(isInIOThread())
        reactor.stop()
    reactor = self.buildReactor()
    event = win32event.CreateEvent(None, False, False, None)
    finished = Deferred()
    listener = Listener(finished)
    finished.addCallback(check)
    reactor.addEvent(event, listener, 'occurred')
    reactor.callWhenRunning(win32event.SetEvent, event)
    self.runReactor(reactor)
    self.assertTrue(listener.success)
    self.assertEqual([True], results)