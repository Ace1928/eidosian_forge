from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_allUnset(self) -> None:
    """
        In an empty L{base.PositionError} with no invariant testing, all
        dilutions of positions are L{None}.
        """
    positionError = base.PositionError()
    self.assertIsNone(positionError.pdop)
    self.assertIsNone(positionError.hdop)
    self.assertIsNone(positionError.vdop)