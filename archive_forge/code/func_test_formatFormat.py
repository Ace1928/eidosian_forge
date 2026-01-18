from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatFormat(self) -> None:
    """
        Formatted event is last field.
        """
    event = dict(log_format='id:{id}', id='123')
    self.assertEqual(formatEventAsClassicLogText(event), '- [-#-] id:123\n')