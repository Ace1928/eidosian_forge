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
def test_foundOnEarlierObject(self) -> None:
    """
        The value returned by L{acquireAttribute} is the value of the requested
        attribute on the first object in the list passed in which has that
        attribute.
        """
    self.value = value = object()
    self.assertTrue(value is acquireAttribute([self, object()], 'value'))