import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not hasWindowsProactorEventLoopPolicy, 'WindowsProactorEventLoopPolicy only on Windows')
def test_WindowsProactorEventLoopPolicy(self):
    """
        L{AsyncioSelectorReactor} will raise a L{TypeError}
        if L{asyncio.WindowsProactorEventLoopPolicy} is default.
        """
    set_event_loop_policy(WindowsProactorEventLoopPolicy())
    self.addCleanup(lambda: set_event_loop_policy(None))
    with self.assertRaises(TypeError):
        AsyncioSelectorReactor()