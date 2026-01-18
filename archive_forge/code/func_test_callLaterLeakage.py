from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_callLaterLeakage(self) -> None:
    """
        callLater should not leak global state into CoreFoundation which will
        be invoked by a different reactor running the main loop.

        @note: this test may actually be usable for other reactors as well, so
            we may wish to promote it to ensure this invariant across other
            foreign-main-loop reactors.
        """
    r = self.buildReactor()
    delayed = r.callLater(0, noop)
    r2 = self.buildReactor()

    def stopBlocking() -> None:
        r2.callLater(0, r2stop)

    def r2stop() -> None:
        r2.stop()
    r2.callLater(0, stopBlocking)
    self.runReactor(r2)
    self.assertEqual(r.getDelayedCalls(), [delayed])