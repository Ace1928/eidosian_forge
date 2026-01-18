from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatTimePercentF(self) -> None:
    """
        "%f" supported in time format.
        """
    self.assertEqual(formatTime(1000000.23456, timeFormat='%f'), '234560')