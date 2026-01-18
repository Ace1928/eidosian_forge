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
def test_failuresAppendTracebacks(self) -> None:
    """
        The string resulting from a logged failure contains a traceback.
        """
    f = Failure(Exception('this is not the behavior you are looking for'))
    log = Logger(observer=self.publisher)
    log.failure('a failure', failure=f)
    msg = self.errorStream.getvalue()
    self.assertIn('a failure', msg)
    self.assertIn('this is not the behavior you are looking for', msg)
    self.assertIn('Traceback', msg)