import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not hasWindowsProactorEventLoopPolicy, 'WindowsProactorEventLoop not available')
def test_WindowsProactorEventLoop(self):
    """
        L{AsyncioSelectorReactor} will raise a L{TypeError}
        if instantiated with a L{asyncio.WindowsProactorEventLoop}
        """
    event_loop = self.newLoop(WindowsProactorEventLoopPolicy())
    self.assertRaises(TypeError, AsyncioSelectorReactor, event_loop)