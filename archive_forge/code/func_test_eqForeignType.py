import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_eqForeignType(self):
    """
        == comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        returns L{NotImplemented}.
        """
    self.assertFalse(SerialNumber(1) == object())
    self.assertIs(SerialNumber(1).__eq__(object()), NotImplemented)