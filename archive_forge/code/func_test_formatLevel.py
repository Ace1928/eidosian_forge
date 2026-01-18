from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatLevel(self) -> None:
    """
        Level is second part of second field.
        """
    event = dict(log_format='XYZZY', log_level=LogLevel.warn)
    self.assertEqual(formatEventAsClassicLogText(event), '- [-#warn] XYZZY\n')