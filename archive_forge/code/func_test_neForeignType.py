import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_neForeignType(self):
    """
        != comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        returns L{NotImplemented}.
        """
    self.assertTrue(SerialNumber(1) != object())
    self.assertIs(SerialNumber(1).__ne__(object()), NotImplemented)