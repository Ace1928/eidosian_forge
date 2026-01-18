from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEventNonCritical(self) -> None:
    """
        An event with no C{log_failure} key will not have a traceback appended.
        """
    event: LogEvent = {'log_format': 'This is a test log message'}
    eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
    self.assertIsInstance(eventText, str)
    self.assertIn('This is a test log message', eventText)