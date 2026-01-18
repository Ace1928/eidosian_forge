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
def test_criticalLogging(self) -> None:
    """
        Critical messages will be written as text to the error stream.
        """
    log = Logger(observer=self.publisher)
    log.info('ignore this')
    log.critical('a critical {message}', message='message')
    self.assertEqual(self.errorStream.getvalue(), 'a critical message\n')