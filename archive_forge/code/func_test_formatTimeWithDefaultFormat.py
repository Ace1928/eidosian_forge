from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatTimeWithDefaultFormat(self) -> None:
    """
        Default time stamp format is RFC 3339 and offset respects the timezone
        as set by the standard C{TZ} environment variable and L{tzset} API.
        """
    if tzset is None:
        raise SkipTest('Platform cannot change timezone; unable to verify offsets.')

    def testForTimeZone(name: str, expectedDST: Optional[str], expectedSTD: str) -> None:
        setTZ(name)
        localSTD = mktime((2007, 1, 31, 0, 0, 0, 2, 31, 0))
        self.assertEqual(formatTime(localSTD), expectedSTD)
        if expectedDST:
            localDST = mktime((2006, 6, 30, 0, 0, 0, 4, 181, 1))
            self.assertEqual(formatTime(localDST), expectedDST)
    testForTimeZone('UTC+00', None, '2007-01-31T00:00:00+0000')
    testForTimeZone('EST+05EDT,M4.1.0,M10.5.0', '2006-06-30T00:00:00-0400', '2007-01-31T00:00:00-0500')
    testForTimeZone('CEST-01CEDT,M4.1.0,M10.5.0', '2006-06-30T00:00:00+0200', '2007-01-31T00:00:00+0100')
    testForTimeZone('CST+06', None, '2007-01-31T00:00:00-0600')