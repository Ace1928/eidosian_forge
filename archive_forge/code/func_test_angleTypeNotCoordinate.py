from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_angleTypeNotCoordinate(self) -> None:
    """
        Creating coordinates with angle types that aren't coordinates raises
        C{ValueError}.
        """
    self.assertRaises(ValueError, base.Coordinate, 150.0, Angles.HEADING)