import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_addOutOfRangeHigh(self):
    """
        L{SerialNumber} cannot be added with other SerialNumber values larger
        than C{_maxAdd}.
        """
    maxAdd = SerialNumber(1)._maxAdd
    self.assertRaises(ArithmeticError, lambda: SerialNumber(1) + SerialNumber(maxAdd + 1))