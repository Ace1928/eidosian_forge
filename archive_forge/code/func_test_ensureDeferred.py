import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_ensureDeferred(self) -> None:
    """
        L{ensureDeferred} will turn a coroutine into a L{Deferred}.
        """

    def run():
        d = succeed('foo')
        res = (yield from d)
        return res
    r = run()
    self.assertIsInstance(r, types.GeneratorType)
    d = ensureDeferred(r)
    self.assertIsInstance(d, Deferred)
    res = self.successResultOf(d)
    self.assertEqual(res, 'foo')