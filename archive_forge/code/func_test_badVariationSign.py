from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_badVariationSign(self) -> None:
    """
        Setting a bogus sign value (not -1 or 1) on a coordinate raises
        C{ValueError} and doesn't affect the coordinate.
        """
    value = 50.0
    c = base.Coordinate(value, Angles.LATITUDE)
    self.assertRaises(ValueError, c.setSign, -50)
    self.assertEqual(c.inDecimalDegrees, 50.0)
    self.assertRaises(ValueError, c.setSign, 0)
    self.assertEqual(c.inDecimalDegrees, 50.0)
    self.assertRaises(ValueError, c.setSign, 50)
    self.assertEqual(c.inDecimalDegrees, 50.0)