import io
from typing import IO, Any, List, Optional, TextIO, Tuple, Type, cast
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._file import textFileLogObserver
from .._global import MORE_THAN_ONCE_WARNING, LogBeginner
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
from ..test.test_stdlib import nextLine
def test_beginLoggingToBufferedEvents(self) -> None:
    """
        Test that events are buffered until C{beginLoggingTo()} is
        called.
        """
    event = dict(foo=1, bar=2)
    events1: List[LogEvent] = []
    events2: List[LogEvent] = []
    o1 = cast(ILogObserver, lambda e: events1.append(e))
    o2 = cast(ILogObserver, lambda e: events2.append(e))
    self.publisher(event)
    self.beginner.beginLoggingTo((o1, o2))
    self.assertEqual([event], events1)
    self.assertEqual([event], events2)