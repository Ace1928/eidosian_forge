import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_maxadd(self):
    """
        In this space, the largest integer that it is meaningful to add to a
        sequence number is 2^(SERIAL_BITS - 1) - 1, or 127.
        """
    self.assertEqual(SerialNumber(0, serialBits=8)._maxAdd, 127)