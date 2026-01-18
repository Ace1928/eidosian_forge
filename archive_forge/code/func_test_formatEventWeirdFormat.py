from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEventWeirdFormat(self) -> None:
    """
        Formatting an event with a bogus format.
        """
    event = dict(log_format=object(), foo=1, bar=2)
    result = formatEvent(event)
    self.assertIn('Log format must be str', result)
    self.assertIn(repr(event), result)