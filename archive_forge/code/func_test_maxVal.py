import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_maxVal(self):
    """
        L{SerialNumber.__add__} returns a wrapped value when s1 plus the s2
        would result in a value greater than the C{maxVal}.
        """
    s = SerialNumber(1)
    maxVal = s._halfRing + s._halfRing - 1
    maxValPlus1 = maxVal + 1
    self.assertTrue(SerialNumber(maxValPlus1) > SerialNumber(maxVal))
    self.assertEqual(SerialNumber(maxValPlus1), SerialNumber(0))