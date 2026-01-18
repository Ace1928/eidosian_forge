from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_eventAsTextSystemMissingNamespaceAndLevel(self) -> None:
    """
        If includeSystem is specified with a missing system but
        namespace and level are present they are used.
        """
    try:
        raise CapturedError('This is a fake error')
    except CapturedError:
        f = Failure()
    t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
    event: LogEvent = {'log_format': 'ABCD', 'log_time': t, 'log_level': LogLevel.info, 'log_namespace': 'test'}
    event['log_failure'] = f
    eventText = eventAsText(event, includeTimestamp=False, includeTraceback=False, includeSystem=True)
    self.assertEqual(eventText, '[test#info] ABCD')