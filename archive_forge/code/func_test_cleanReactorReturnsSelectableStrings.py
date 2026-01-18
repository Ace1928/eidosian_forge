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
def test_cleanReactorReturnsSelectableStrings(self) -> None:
    """
        The Janitor returns string representations of the selectables that it
        cleaned up from the reactor cleanup method.
        """

    class Selectable:
        """
            A stub Selectable which only has an interesting string
            representation.
            """

        def __repr__(self) -> str:
            return '(SELECTABLE!)'
    reactor = StubReactor([], [Selectable()])
    jan = _Janitor(None, None, reactor=reactor)
    self.assertEqual(jan._cleanReactor(), ['(SELECTABLE!)'])