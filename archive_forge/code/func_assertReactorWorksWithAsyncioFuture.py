import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def assertReactorWorksWithAsyncioFuture(self, reactor):
    """
        Ensure that C{reactor} has an event loop that works
        properly with L{asyncio.Future}.
        """
    future = Future()
    result = []

    def completed(future):
        result.append(future.result())
        reactor.stop()
    future.add_done_callback(completed)
    future.set_result(True)
    self.assertEqual(result, [])
    self.runReactor(reactor, timeout=1)
    self.assertEqual(result, [True])