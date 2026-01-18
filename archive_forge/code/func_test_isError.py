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
def test_isError(self) -> None:
    """
        If C{"isError"} is set to C{1} (true) on the legacy event, the
        C{"log_level"} key should get set to L{LogLevel.critical}.
        """
    publishToNewObserver(self.observer, self.legacyEvent(isError=1), legacyLog.textFromEventDict)
    self.assertEqual(self.events[0]['log_level'], LogLevel.critical)