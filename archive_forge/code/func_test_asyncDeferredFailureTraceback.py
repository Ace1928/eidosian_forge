import types
from typing_extensions import NoReturn
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_asyncDeferredFailureTraceback(self) -> None:
    """
        When a Deferred is awaited upon that later fails with a Failure that
        has a traceback, both the place that the synchronous traceback comes
        from and the awaiting line are shown in the traceback.
        """

    def returnsFailure() -> Failure:
        try:
            raise SampleException()
        except SampleException:
            return Failure()
    it: Deferred[None] = Deferred()

    async def doomed() -> None:
        return await it
    started = Deferred.fromCoroutine(doomed())
    self.assertNoResult(started)
    it.errback(returnsFailure())
    failure = self.failureResultOf(started)
    self.assertIn(', in doomed\n', failure.getTraceback())
    self.assertIn(', in returnsFailure\n', failure.getTraceback())