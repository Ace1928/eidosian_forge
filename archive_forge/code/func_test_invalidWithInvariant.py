from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_invalidWithInvariant(self) -> None:
    """
        Creating a L{base.PositionError} with values set to an impossible
        combination raises C{ValueError} if the invariant is being tested.
        """
    self.assertRaises(ValueError, base.PositionError, pdop=1.0, vdop=1.0, hdop=1.0, testInvariant=True)