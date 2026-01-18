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
def test_criticalLoggingStops(self) -> None:
    """
        Once logging has begun with C{beginLoggingTo}, critical messages are no
        longer written to the output stream.
        """
    log = Logger(observer=self.publisher)
    self.beginner.beginLoggingTo(())
    log.critical('another critical message')
    self.assertEqual(self.errorStream.getvalue(), '')