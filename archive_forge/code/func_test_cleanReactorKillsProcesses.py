from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_cleanReactorKillsProcesses(self) -> None:
    """
        The Janitor will kill processes during reactor cleanup.
        """

    @implementer(IProcessTransport)
    class StubProcessTransport:
        """
            A stub L{IProcessTransport} provider which records signals.
            @ivar signals: The signals passed to L{signalProcess}.
            """

        def __init__(self) -> None:
            self.signals: list[str | int] = []

        def signalProcess(self, signal: str | int) -> None:
            """
                Append C{signal} to C{self.signals}.
                """
            self.signals.append(signal)
    pt = StubProcessTransport()
    reactor = StubReactor([], [pt])
    jan = _Janitor(None, None, reactor=reactor)
    jan._cleanReactor()
    self.assertEqual(pt.signals, ['KILL'])