from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_latitudeTooSmall(self) -> None:
    """
        Creating a latitude with a value less than or equal to -90 degrees
        raises C{ValueError}.
        """
    self.assertRaises(ValueError, _makeLatitude, -150.0)
    self.assertRaises(ValueError, _makeLatitude, -90.0)