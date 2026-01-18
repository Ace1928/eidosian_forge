import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not _defaultEventLoopIsSelector, 'default event loop: {}\nis not of type SelectorEventLoop on Python {}.{} ({})'.format(type(_defaultEventLoop), sys.version_info.major, sys.version_info.minor, platform.getType()))
def test_defaultSelectorEventLoopFromGlobalPolicy(self):
    """
        L{AsyncioSelectorReactor} wraps the global policy's event loop
        by default.  This ensures that L{asyncio.Future}s and
        coroutines created by library code that uses
        L{asyncio.get_event_loop} are bound to the same loop.
        """
    reactor = AsyncioSelectorReactor()
    self.assertReactorWorksWithAsyncioFuture(reactor)