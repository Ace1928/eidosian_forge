from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_longitudeTooSmall(self) -> None:
    """
        Creating a longitude with a value less than or equal to -180 degrees
        raises C{ValueError}.
        """
    self.assertRaises(ValueError, _makeLongitude, -250.0)
    self.assertRaises(ValueError, _makeLongitude, -180.0)