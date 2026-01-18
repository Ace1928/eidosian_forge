import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not hasWindowsSelectorEventLoopPolicy, 'WindowsSelectorEventLoop only on Windows')
def test_WindowsSelectorEventLoop(self):
    """
        L{WindowsSelectorEventLoop} works with L{AsyncioSelectorReactor}
        """
    event_loop = self.newLoop(WindowsSelectorEventLoopPolicy())
    reactor = AsyncioSelectorReactor(event_loop)
    set_event_loop(event_loop)
    self.assertReactorWorksWithAsyncioFuture(reactor)