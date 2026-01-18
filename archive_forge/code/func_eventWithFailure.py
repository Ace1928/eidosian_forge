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
def eventWithFailure(self, **values: object) -> LogEvent:
    """
        Create a new-style event with a captured failure.

        @param values: Additional values to include in the event.

        @return: the new event
        """
    failure = Failure(RuntimeError('nyargh!'))
    return self.forwardAndVerify(dict(log_failure=failure, log_format='oopsie...', **values))