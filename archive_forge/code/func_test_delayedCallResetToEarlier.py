import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_delayedCallResetToEarlier(self):
    """
        L{DelayedCall.reset()} properly reschedules timer to earlier time
        """
    if hasWindowsSelectorEventLoopPolicy:
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    reactor = AsyncioSelectorReactor()
    timer_called_at = [None]

    def on_timer():
        timer_called_at[0] = reactor.seconds()
    start_time = reactor.seconds()
    dc = reactor.callLater(0.5, on_timer)
    dc.reset(0)
    reactor.callLater(1, reactor.stop)
    import io
    from contextlib import redirect_stderr
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        reactor.run()
    self.assertEqual(stderr.getvalue(), '')
    self.assertIsNotNone(timer_called_at[0])
    self.assertLess(timer_called_at[0] - start_time, 0.4)
    if hasWindowsSelectorEventLoopPolicy:
        set_event_loop_policy(None)