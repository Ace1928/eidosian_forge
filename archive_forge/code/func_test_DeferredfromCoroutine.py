import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_DeferredfromCoroutine(self) -> None:
    """
        L{Deferred.fromCoroutine} will turn a coroutine into a L{Deferred}.
        """

    def run():
        d = succeed('bar')
        yield from d
        res = (yield from run2())
        return res

    def run2():
        d = succeed('foo')
        res = (yield from d)
        return res
    r = run()
    self.assertIsInstance(r, types.GeneratorType)
    d = Deferred.fromCoroutine(r)
    self.assertIsInstance(d, Deferred)
    res = self.successResultOf(d)
    self.assertEqual(res, 'foo')