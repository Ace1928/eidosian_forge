import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_noCycleReferencesInCallLater(self):
    """
        L{AsyncioSelectorReactor.callLater()} doesn't leave cyclic references
        """
    if hasWindowsSelectorEventLoopPolicy:
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        objects_before = len(gc.get_objects())
        timer_count = 1000
        reactor = AsyncioSelectorReactor()
        for _ in range(timer_count):
            reactor.callLater(0, lambda: None)
        reactor.runUntilCurrent()
        objects_after = len(gc.get_objects())
        self.assertLess((objects_after - objects_before) / timer_count, 1)
    finally:
        if gc_was_enabled:
            gc.enable()
    if hasWindowsSelectorEventLoopPolicy:
        set_event_loop_policy(None)