import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not _defaultEventLoopIsSelector, 'default event loop: {}\nis not of type SelectorEventLoop on Python {}.{} ({})'.format(type(_defaultEventLoop), sys.version_info.major, sys.version_info.minor, platform.getType()))
def test_newSelectorEventLoopFromDefaultEventLoopPolicy(self):
    """
        If we use the L{asyncio.DefaultLoopPolicy} to create a new event loop,
        and then pass that event loop to a new L{AsyncioSelectorReactor},
        this reactor should work properly with L{asyncio.Future}.
        """
    event_loop = self.newLoop(DefaultEventLoopPolicy())
    reactor = AsyncioSelectorReactor(event_loop)
    set_event_loop(event_loop)
    self.assertReactorWorksWithAsyncioFuture(reactor)