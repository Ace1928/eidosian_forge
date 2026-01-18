import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_ltForeignType(self):
    """
        < comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        raises L{TypeError}.
        """
    self.assertRaises(TypeError, lambda: SerialNumber(1) < object())