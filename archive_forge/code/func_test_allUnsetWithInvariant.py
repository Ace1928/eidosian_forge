from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_allUnsetWithInvariant(self) -> None:
    """
        In an empty L{base.PositionError} with invariant testing, all
        dilutions of positions are L{None}.
        """
    positionError = base.PositionError(testInvariant=True)
    self.assertIsNone(positionError.pdop)
    self.assertIsNone(positionError.hdop)
    self.assertIsNone(positionError.vdop)