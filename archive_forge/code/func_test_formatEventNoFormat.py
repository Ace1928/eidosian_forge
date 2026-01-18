from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEventNoFormat(self) -> None:
    """
        Formatting an event with no format.
        """
    event = dict(foo=1, bar=2)
    result = formatEvent(event)
    self.assertEqual('', result)