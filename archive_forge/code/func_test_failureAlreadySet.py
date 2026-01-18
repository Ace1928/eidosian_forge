import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
def test_failureAlreadySet(self) -> None:
    """
        Captured failures in the new style do not step on a pre-existing
        old-style C{"failure"} key.
        """
    failure = Failure(RuntimeError('Weak salsa!'))
    event = self.eventWithFailure(failure=failure)
    self.assertIs(event['failure'], failure)