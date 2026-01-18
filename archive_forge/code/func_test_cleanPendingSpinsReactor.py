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
def test_cleanPendingSpinsReactor(self) -> None:
    """
        During pending-call cleanup, the reactor will be spun twice with an
        instant timeout. This is not a requirement, it is only a test for
        current behavior. Hopefully Trial will eventually not do this kind of
        reactor stuff.
        """
    reactor = StubReactor([])
    jan = _Janitor(None, None, reactor=reactor)
    jan._cleanPending()
    self.assertEqual(reactor.iterations, [0, 0])