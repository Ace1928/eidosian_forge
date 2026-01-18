import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_convertOtherSerialBitsMismatch(self):
    """
        L{SerialNumber._convertOther} raises L{TypeError} if the other
        SerialNumber instance has a different C{serialBits} value.
        """
    s1 = SerialNumber(0, serialBits=8)
    s2 = SerialNumber(0, serialBits=16)
    self.assertRaises(TypeError, s1._convertOther, s2)