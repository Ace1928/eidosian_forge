from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatUnformattableErrorWithTraceback(self) -> None:
    """
        An event with an unformattable value in the C{log_format} key, that
        throws an exception when __repr__ is invoked still has a traceback
        appended.
        """
    try:
        raise CapturedError('This is a fake error')
    except CapturedError:
        f = Failure()
    event: LogEvent = {'log_format': '{evil()}', 'evil': lambda: 1 / 0, cast(str, Unformattable()): 'gurk'}
    event['log_failure'] = f
    eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
    self.assertIsInstance(eventText, str)
    self.assertIn('MESSAGE LOST', eventText)
    self.assertIn(str(f.getTraceback()), eventText)
    self.assertIn('This is a fake error', eventText)