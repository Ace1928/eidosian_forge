from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEmptyFormat(self) -> None:
    """
        Empty format string.
        """
    event = dict(log_format='', id='123')
    self.assertIs(formatEventAsClassicLogText(event), None)