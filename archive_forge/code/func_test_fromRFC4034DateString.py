import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_fromRFC4034DateString(self):
    """
        L{SerialNumber.fromRFC4034DateString} accepts a datetime string argument
        of the form 'YYYYMMDDhhmmss' and returns an L{SerialNumber} instance
        whose value is the unix timestamp corresponding to that UTC date.
        """
    self.assertEqual(SerialNumber(1325376000), SerialNumber.fromRFC4034DateString('20120101000000'))