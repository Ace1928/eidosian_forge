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
def test_beginLoggingToDontRedirect(self) -> None:
    """
        L{LogBeginner.beginLoggingTo} will leave the existing stdout/stderr in
        place if it has been told not to replace them.
        """
    oldOut = self.sysModule.stdout
    oldErr = self.sysModule.stderr
    self.beginner.beginLoggingTo((), redirectStandardIO=False)
    self.assertIs(self.sysModule.stdout, oldOut)
    self.assertIs(self.sysModule.stderr, oldErr)