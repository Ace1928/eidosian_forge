from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testStopAtOnceLater(self):
    d = defer.Deferred()

    def foo():
        d.errback(failure.DefaultException('This task also should never get called.'))
    self._lc = task.LoopingCall(foo)
    self._lc.start(1, now=False)
    reactor.callLater(0, self._callback_for_testStopAtOnceLater, d)
    return d