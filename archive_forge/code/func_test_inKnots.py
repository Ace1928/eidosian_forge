from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_inKnots(self) -> None:
    """
        A speed can be converted into its value in knots.
        """
    speed = base.Speed(1.0)
    self.assertEqual(1 / base.MPS_PER_KNOT, speed.inKnots)