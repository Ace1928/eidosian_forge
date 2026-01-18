from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_differentAnglesInequality(self) -> None:
    """
        Coordinates with different values aren't equal.
        """
    c1 = base.Coordinate(1.0)
    c2 = base.Coordinate(-1.0)
    self.assertNotEqual(c1, c2)