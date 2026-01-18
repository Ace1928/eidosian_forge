import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_chained(self) -> None:
    """
        Yielding from a paused & chained Deferred will give the result when it
        has one.
        """
    reactor = Clock()

    def test():
        d = Deferred()
        d2 = Deferred()
        d.addCallback(lambda ignored: d2)
        d.callback(None)
        reactor.callLater(0, d2.callback, 'bye')
        res = (yield from d)
        return res
    d = Deferred.fromCoroutine(test())
    reactor.advance(0.1)
    res = self.successResultOf(d)
    self.assertEqual(res, 'bye')