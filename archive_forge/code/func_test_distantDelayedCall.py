from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def test_distantDelayedCall(self):
    """
        Scheduling a delayed call at a point in the extreme future does not
        prevent normal reactor operation.
        """
    reactor = self.buildReactor()
    if IReactorThreads.providedBy(reactor):

        def eventSource(reactor, event):
            msg(format='Thread-based event-source scheduling %(event)r', event=event)
            reactor.callFromThread(event)
    else:
        raise SkipTest('Do not know how to synthesize non-time event to stop the test')
    delayedCall = reactor.callLater(2 ** 128 + 1, lambda: None)

    def stop():
        msg('Stopping the reactor')
        reactor.stop()
    eventSource(reactor, lambda: eventSource(reactor, stop))
    reactor.run()
    self.assertTrue(delayedCall.active())
    self.assertIn(delayedCall, reactor.getDelayedCalls())